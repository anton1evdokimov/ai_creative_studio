#!/usr/bin/env python3
"""Train a small SDXL UNet LoRA from a folder of images.

Train against SDXL *base* (not Turbo). Then enable the weights in config.yaml:

    diffusion:
      model: sdxl
      lora:
        enabled: true
        path: lora/xyz_hoodie
        scale: 0.8
        trigger: "xyz hoodie"

Example:

    python scripts/train_sdxl_lora.py \\
      --instance_data_dir data/lora/hoodies \\
      --instance_prompt "a photo of xyz hoodie" \\
      --output_dir lora/xyz_hoodie \\
      --max_train_steps 500 --rank 8 --resolution 512 \\
      --eval_every 100

Watch CLIP-T / CLIP-I in output_dir/metrics.csv (not train MSE) to pick a checkpoint.
Optional: --eval_vlm (heavy, skip on 8GB) and --eval_ocr (needs pytesseract + tesseract).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        try:
            if mps.is_available() or mps.is_built():
                return "mps"
        except Exception:
            return "mps"
    return "cpu"


class ImageFolderCaption(Dataset):
    def __init__(self, root: Path, caption: str, size: int):
        self.paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images in {root}")
        self.caption = caption
        self.tf = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"pixel_values": self.tf(img), "caption": self.caption}


def encode_prompt(caption: str, tokenizers, text_encoders, device, dtype):
    prompt_embeds_list = []
    pooled = None
    for tokenizer, encoder in zip(tokenizers, text_encoders):
        tokens = tokenizer(
            caption,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out = encoder(
            tokens.input_ids.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        prompt_embeds_list.append(out.hidden_states[-2].to(dtype=dtype))
        pooled = out[0].to(dtype=dtype)
    return torch.cat(prompt_embeds_list, dim=-1), pooled


def time_ids(resolution: int, batch: int, device, dtype) -> torch.Tensor:
    ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        device=device,
        dtype=dtype,
    )
    return ids.unsqueeze(0).repeat(batch, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Train SDXL LoRA (UNet only)")
    p.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--instance_data_dir", required=True)
    p.add_argument("--instance_prompt", required=True)
    p.add_argument("--output_dir", default="lora/run")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_steps", type=int, default=500)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpointing_steps", type=int, default=100)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    p.add_argument("--eval_every", type=int, default=100, help="Generate+score every N steps (0=off)")
    p.add_argument("--eval_prompt", default="", help="Prompt for eval images; default = instance_prompt")
    p.add_argument("--eval_num_images", type=int, default=1)
    p.add_argument("--eval_inference_steps", type=int, default=8)
    p.add_argument("--eval_guidance", type=float, default=5.0)
    p.add_argument("--eval_vlm", action="store_true", help="Also run VLM-as-judge (heavy)")
    p.add_argument("--eval_ocr", action="store_true", help="OCR generated image (pytesseract)")
    p.add_argument("--ocr_terms", default="", help="Comma-separated words that should appear on the product")
    return p.parse_args()


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def ocr_image(image: Image.Image) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(image) or ""
    except Exception:
        return ""


def ocr_hit_rate(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    blob = text.lower()
    hits = sum(1 for t in terms if t and t.lower() in blob)
    return round(hits / max(1, len(terms)), 3)


def _ocr_terms(args) -> list[str]:
    terms = [t.strip() for t in (args.ocr_terms or "").split(",") if t.strip()]
    if terms:
        return terms
    prompt = (args.eval_prompt or args.instance_prompt).strip()
    stop = {"a", "an", "the", "of", "photo", "image", "picture", "with", "and", "on", "in"}
    return [w for w in prompt.replace(",", " ").split() if len(w) > 3 and w.lower() not in stop][:6]


def run_eval(
    *,
    args,
    step: int,
    loss: float,
    out_dir: Path,
    train_image_paths: list[Path],
    vae,
    unet,
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    pretrained: str,
) -> dict:
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

    unet.eval()
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    prompt = (args.eval_prompt or args.instance_prompt).strip()
    terms = _ocr_terms(args)
    refs = train_image_paths[:4]

    scheduler = EulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
    )
    pipe.set_progress_bar_config(disable=True)

    saved: list[tuple[Path, Image.Image]] = []
    with torch.no_grad():
        for i in range(max(1, args.eval_num_images)):
            image = pipe(
                prompt=prompt,
                num_inference_steps=int(args.eval_inference_steps),
                guidance_scale=float(args.eval_guidance),
                height=int(args.resolution),
                width=int(args.resolution),
                generator=torch.Generator(device="cpu").manual_seed(args.seed + step + i),
            ).images[0]
            img_path = eval_dir / f"step{step:05d}_{i}.png"
            image.save(img_path)
            saved.append((img_path, image))

    del pipe
    unet.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clipper = None
    vlm_scorer = None
    clip_error = ""
    try:
        from models.diffusion.config import load_ranking_config
        from models.ranking.clip_metrics import CLIPMetrics

        rcfg = load_ranking_config()
        # CLIP on CPU: SDXL still occupies the GPU during train eval
        clipper = CLIPMetrics(
            rcfg.get("clip_model", "openai/clip-vit-large-patch14"),
            use_aesthetic=True,
            device="cpu",
        )
        print("   CLIP eval on cpu (so train GPU stays free)")
    except Exception as exc:
        clip_error = f"{type(exc).__name__}: {exc}"
        print(f"⚠️  CLIP eval skipped: {clip_error}")
    if args.eval_vlm:
        try:
            from models.ranking.clip import VLMImageScorer

            vlm_scorer = VLMImageScorer()
        except Exception as exc:
            print(f"⚠️  VLM judge skipped: {exc}")

    clip_ts, clip_is, aesths, vlm_over, ocr_hits = [], [], [], [], []
    clip_t_raws, clip_i_raws = [], []
    dino_is, dino_raws = [], []
    last_ocr = ""
    for img_path, image in saved:
        if clipper is not None:
            try:
                clip_m = clipper.score(img_path, prompt, product_image_path=refs[0] if refs else None)
                clip_ts.append(clip_m.clip_t)
                clip_t_raws.append(clip_m.clip_t_raw)
                aesths.append(clip_m.aesthetic)
                i_vals, i_raws = [], []
                for ref in refs:
                    m = clipper.score(img_path, prompt, product_image_path=ref)
                    if m.clip_i is not None:
                        i_vals.append(m.clip_i)
                        if m.clip_i_raw is not None:
                            i_raws.append(m.clip_i_raw)
                if i_vals:
                    clip_is.append(sum(i_vals) / len(i_vals))
                if i_raws:
                    clip_i_raws.append(sum(i_raws) / len(i_raws))
            except Exception as exc:
                clip_error = f"{type(exc).__name__}: {exc}"
                print(f"⚠️  CLIP score failed on {img_path.name}: {clip_error}")
        if vlm_scorer is not None:
            vlm = vlm_scorer.score(img_path, prompt_text=prompt, context_text=prompt)
            vlm_over.append(vlm.overall)
        if args.eval_ocr:
            last_ocr = ocr_image(image)
            ocr_hits.append(ocr_hit_rate(last_ocr, terms))
        print(
            f"   eval {img_path.name}: "
            f"CLIP-T={clip_ts[-1] if clip_ts else '—'} "
            f"(raw={clip_t_raws[-1] if clip_t_raws else '—'}) "
            f"CLIP-I={round(clip_is[-1], 4) if clip_is else '—'}"
        )

    try:
        from models.ranking.clip_metrics import unload_clip_metrics
        from models.ranking.clip import reset_image_scorer

        unload_clip_metrics()
        reset_image_scorer()
    except Exception:
        pass

    dino_error = ""
    try:
        from models.diffusion.config import load_ranking_config
        from models.ranking.dino_metrics import DinoMetrics, unload_dino_metrics

        rcfg = load_ranking_config()
        if refs:
            dino = DinoMetrics(rcfg.get("dino_model") or "facebook/dinov2-small", device="cpu")
            print("   DINOv2 eval on cpu")
            for img_path, _image in saved:
                s, raw = dino.similarity_mean(img_path, refs)
                if s is not None:
                    dino_is.append(s)
                    dino_raws.append(raw)
                    print(f"   eval {img_path.name}: DINO-I={s:.3f} (cos={raw:.3f})")
            unload_dino_metrics()
    except Exception as exc:
        dino_error = f"{type(exc).__name__}: {exc}"
        print(f"⚠️  DINOv2 eval skipped: {dino_error}")

    row = {
        "step": step,
        "loss": round(loss, 6),
        "clip_t": round(sum(clip_ts) / len(clip_ts), 4) if clip_ts else "",
        "clip_t_raw": round(sum(clip_t_raws) / len(clip_t_raws), 4) if clip_t_raws else "",
        "clip_i": round(sum(clip_is) / len(clip_is), 4) if clip_is else "",
        "clip_i_raw": round(sum(clip_i_raws) / len(clip_i_raws), 4) if clip_i_raws else "",
        "dino_i": round(sum(dino_is) / len(dino_is), 4) if dino_is else "",
        "dino_i_raw": round(sum(dino_raws) / len(dino_raws), 4) if dino_raws else "",
        "aesthetic": round(sum(aesths) / len(aesths), 4) if aesths else "",
        "vlm_overall": round(sum(vlm_over) / len(vlm_over), 4) if vlm_over else "",
        "ocr_hit": round(sum(ocr_hits) / len(ocr_hits), 4) if ocr_hits else "",
        "ocr_text": " ".join(last_ocr.split())[:180],
        "clip_error": (clip_error or dino_error)[:200],
        "prompt": prompt,
    }
    append_csv(out_dir / "metrics.csv", row)
    print(f"   metrics → {out_dir / 'metrics.csv'}: {row}")
    return row


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    lora_alpha = args.lora_alpha or args.rank
    data_dir = Path(args.instance_data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device == "mps" and args.mixed_precision != "no":
        print("⚠️  Mixed precision on MPS is unstable — using float32")
        args.mixed_precision = "no"

    dtype = torch.float32
    if args.mixed_precision == "fp16" and device == "cuda":
        dtype = torch.float16
    elif args.mixed_precision == "bf16" and device == "cuda":
        dtype = torch.bfloat16

    print(f"Device={device}  rank={args.rank}  steps={args.max_train_steps}")
    print(f"Data={data_dir}  prompt={args.instance_prompt!r}")

    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    tokenizer_1 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    text_encoder_1 = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device, dtype=dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2").to(device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device, dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device, dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder_1.eval()
    text_encoder_2.eval()

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    unet.train()
    unet.enable_gradient_checkpointing()

    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    dataset = ImageFolderCaption(data_dir, args.instance_prompt, args.resolution)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    print(f"Images: {len(dataset)}")
    if args.eval_every > 0:
        print(f"Eval every {args.eval_every} steps → {out_dir / 'metrics.csv'} (CLIP-T / CLIP-I)")
        if args.eval_vlm:
            print("VLM-as-judge enabled (heavy on 8GB)")
        if args.eval_ocr:
            print("OCR enabled (needs pytesseract + tesseract)")

    tokenizers = [tokenizer_1, tokenizer_2]
    encoders = [text_encoder_1, text_encoder_2]

    global_step = 0
    last_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=args.max_train_steps, desc="lora")

    while global_step < args.max_train_steps:
        for batch in loader:
            pixels = batch["pixel_values"].to(device=device, dtype=torch.float32)
            raw_caps = batch["caption"]
            caption = raw_caps[0] if isinstance(raw_caps, (list, tuple)) else raw_caps

            with torch.no_grad():
                latents = vae.encode(pixels).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=dtype)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long
            )
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                prompt_embeds, pooled = encode_prompt(caption, tokenizers, encoders, device, dtype)
                if prompt_embeds.shape[0] != bsz:
                    prompt_embeds = prompt_embeds.repeat(bsz, 1, 1)
                    pooled = pooled.repeat(bsz, 1)

            add_time = time_ids(args.resolution, bsz, device, dtype)
            pred = unet(
                noisy,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time},
            ).sample
            loss = F.mse_loss(pred.float(), noise.float()) / args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            last_loss = float(loss.item() * args.gradient_accumulation_steps)
            pbar.update(1)
            pbar.set_postfix(loss=f"{last_loss:.4f}")

            if global_step % args.checkpointing_steps == 0 or global_step >= args.max_train_steps:
                ckpt = out_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
                StableDiffusionXLPipeline.save_lora_weights(str(ckpt), unet_lora_layers=unet_lora)
                print(f"\nSaved {ckpt}")

            if args.eval_every > 0 and (
                global_step % args.eval_every == 0 or global_step >= args.max_train_steps
            ):
                run_eval(
                    args=args,
                    step=global_step,
                    loss=last_loss,
                    out_dir=out_dir,
                    train_image_paths=dataset.paths,
                    vae=vae,
                    unet=unet,
                    text_encoder_1=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    tokenizer_1=tokenizer_1,
                    tokenizer_2=tokenizer_2,
                    pretrained=args.pretrained_model_name_or_path,
                )

            if global_step >= args.max_train_steps:
                break

    unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(str(out_dir), unet_lora_layers=unet_lora)
    (out_dir / "trigger.txt").write_text(args.instance_prompt + "\n", encoding="utf-8")
    print(f"\nDone. LoRA weights → {out_dir}")
    print("Enable in config.yaml:")
    print("  diffusion:")
    print("    model: sdxl")
    print("    lora:")
    print("      enabled: true")
    print(f"      path: {out_dir}")
    print("      scale: 0.8")
    print(f"      trigger: {args.instance_prompt!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
