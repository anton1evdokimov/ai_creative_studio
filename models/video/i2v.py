"""Product video: Kandinsky 5 T2V (text-to-video). Ken Burns GIF if T2V fails."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

K5_T2V_DEFAULT = "kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers"


def kenburns_gif(image_path: str | Path, output_path: str | Path, frames: int = 16) -> str:
    src = Image.open(image_path).convert("RGB")
    w, h = src.size
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    seq = []
    for i in range(max(2, frames)):
        t = i / max(frames - 1, 1)
        zoom = 1.0 + 0.08 * t
        cw, ch = max(1, int(w / zoom)), max(1, int(h / zoom))
        left = (w - cw) // 2
        top = int((h - ch) * (0.15 * t))
        crop = src.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.LANCZOS)
        seq.append(crop)
    seq[0].save(out, save_all=True, append_images=seq[1:], duration=80, loop=0)
    return str(out)


def kandinsky_t2v_mp4(
    prompt: str,
    output_path: str | Path,
    *,
    model_id: str = K5_T2V_DEFAULT,
    num_frames: int = 241,
    steps: int = 16,
    guidance: float = 1.0,
    width: int = 768,
    height: int = 512,
) -> str:
    import torch
    from diffusers.utils import export_to_video

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_id = model_id or K5_T2V_DEFAULT
    try:
        from diffusers import Kandinsky5T2VPipeline as PipeCls
    except ImportError as exc:
        raise RuntimeError("Need diffusers with Kandinsky5T2VPipeline") from exc

    ids = [model_id]
    if "Diffusers" not in model_id:
        ids.append(model_id.rstrip("/") + "-Diffusers")
    pipe = None
    last = None
    for mid in ids:
        try:
            print(f"   Loading K5 T2V ({mid})")
            pipe = PipeCls.from_pretrained(mid, torch_dtype=torch.bfloat16)
            break
        except Exception as exc:
            last = exc
    if pipe is None:
        raise RuntimeError(last)
    if torch.cuda.is_available():
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    try:
        pipe.transformer.set_attention_backend("flex")
    except Exception:
        pass
    neg = "static, 2d cartoon, worst quality, low quality, ugly, deformed, text artifacts"
    kwargs = dict(
        prompt=(prompt or "cinematic product commercial, slow camera move")[:400],
        negative_prompt=neg,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
    )
    print(f"   K5 T2V  {width}x{height}  frames={num_frames}  steps={steps}  gs={guidance}")
    frames = pipe(**kwargs).frames[0]
    export_to_video(frames, str(out), fps=24)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(out)


def generate_product_video(
    image_path: str | Path,
    output_path: str | Path,
    *,
    backend: str = "kandinsky5_t2v",
    prompt: str = "",
    model_id: str = K5_T2V_DEFAULT,
    num_frames: int = 241,
    **extra,
) -> str:
    model_id = model_id or K5_T2V_DEFAULT
    out = Path(output_path)
    if backend in {"kandinsky5_t2v", "kandinsky", "k5_t2v", "t2v"}:
        try:
            return kandinsky_t2v_mp4(
                prompt,
                out.with_suffix(".mp4"),
                model_id=model_id,
                num_frames=num_frames,
                steps=int(extra.get("num_inference_steps") or 16),
                guidance=float(extra.get("guidance_scale") or 1.0),
                width=int(extra.get("width") or 768),
                height=int(extra.get("height") or 512),
            )
        except Exception as exc:
            print(f"⚠️  K5 T2V failed ({type(exc).__name__}: {exc}) — Ken Burns GIF")
    if backend == "svd":
        print("⚠️  SVD backend removed — use kandinsky5_t2v")
    return kenburns_gif(image_path, out.with_suffix(".gif"), frames=16)
