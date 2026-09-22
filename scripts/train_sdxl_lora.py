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
      --max_train_steps 500 --rank 8 --resolution 512
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


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
    return p.parse_args()


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

    tokenizers = [tokenizer_1, tokenizer_2]
    encoders = [text_encoder_1, text_encoder_2]

    global_step = 0
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
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}")

            if global_step % args.checkpointing_steps == 0 or global_step >= args.max_train_steps:
                ckpt = out_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
                StableDiffusionXLPipeline.save_lora_weights(str(ckpt), unet_lora_layers=unet_lora)
                print(f"\nSaved {ckpt}")

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
