"""Product video: Kandinsky 5 I2V Lite 5s. Ken Burns GIF if I2V fails."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

K5_I2V_DEFAULT = "kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers"

_MOTION = (
    "smooth cinematic camera motion, slow dolly-in, subtle orbit, "
    "gentle background parallax, product stays sharp and fully in frame"
)


def _letterbox(src: Image.Image, width: int, height: int) -> Image.Image:
    src = src.convert("RGB")
    scale = min(width / src.width, height / src.height)
    nw, nh = max(1, int(src.width * scale)), max(1, int(src.height * scale))
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (8, 8, 8))
    canvas.paste(resized, ((width - nw) // 2, (height - nh) // 2))
    return canvas


def _i2v_hw(src: Image.Image, width: int, height: int) -> tuple[int, int]:
    """Lite I2V: landscape 768x512, portrait 512x768. Do not squash the still."""
    w, h = int(width), int(height)
    if src.height >= src.width and w > h:
        w, h = h, w
    elif src.width > src.height and h > w:
        w, h = h, w
    w = max(256, min(w, 768))
    h = max(256, min(h, 768))
    w -= w % 64
    h -= h % 64
    return max(w, 256), max(h, 256)


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


def kandinsky_i2v_mp4(
    image_path: str | Path,
    prompt: str,
    output_path: str | Path,
    *,
    model_id: str = K5_I2V_DEFAULT,
    num_frames: int = 121,
    steps: int = 50,
    guidance: float = 5.0,
    width: int = 768,
    height: int = 512,
) -> str:
    import torch
    from diffusers.utils import export_to_video
    from models.mem import release_cuda

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_id = model_id or K5_I2V_DEFAULT
    src = Image.open(image_path).convert("RGB")
    width, height = _i2v_hw(src, width, height)
    num_frames = max(9, min(int(num_frames), 121))
    try:
        from diffusers import Kandinsky5I2VPipeline as PipeCls
    except ImportError as exc:
        raise RuntimeError("Need diffusers with Kandinsky5I2VPipeline") from exc

    ids = [model_id]
    if "Diffusers" not in model_id:
        ids.append(model_id.rstrip("/") + "-Diffusers")
    pipe = None
    last = None
    release_cuda()
    for mid in ids:
        try:
            print(f"   Loading K5 I2V ({mid})")
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
    image = _letterbox(src, width, height)
    text = (prompt or "").strip()
    low = text.lower()
    if not any(k in low for k in ("dolly", "orbit", "parallax", "camera motion", "push-in", "push in")):
        text = f"{_MOTION}. {text}" if text else _MOTION
    text = text[:400]
    neg = (
        "Static, freeze frame, no motion, still photograph, 2D cartoon, cartoon, "
        "2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"
    )
    kwargs = dict(
        image=image,
        prompt=text,
        negative_prompt=neg,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
    )
    print(f"   K5 I2V  {Path(image_path).name}  {width}x{height}  frames={num_frames}  steps={steps}  gs={guidance}")
    frames = pipe(**kwargs).frames[0]
    try:
        export_to_video(frames, str(out), fps=24, quality=9)
    except TypeError:
        export_to_video(frames, str(out), fps=24)
    del pipe
    release_cuda()
    return str(out)


def generate_product_video(
    image_path: str | Path,
    output_path: str | Path,
    *,
    backend: str = "kandinsky5_i2v",
    prompt: str = "",
    model_id: str = K5_I2V_DEFAULT,
    num_frames: int = 121,
    **extra,
) -> str:
    model_id = model_id or K5_I2V_DEFAULT
    out = Path(output_path)
    if backend in {"kenburns", "gif"}:
        return kenburns_gif(image_path, out.with_suffix(".gif"), frames=16)
    if backend in {"kandinsky5_i2v", "i2v", "kandinsky5_t2v", "kandinsky", "k5_t2v", "t2v"}:
        try:
            return kandinsky_i2v_mp4(
                image_path,
                prompt,
                out.with_suffix(".mp4"),
                model_id=model_id,
                num_frames=num_frames,
                steps=int(extra.get("num_inference_steps") or 50),
                guidance=float(extra.get("guidance_scale") or 5.0),
                width=int(extra.get("width") or 768),
                height=int(extra.get("height") or 512),
            )
        except Exception as exc:
            print(f"⚠️  K5 I2V failed ({type(exc).__name__}: {exc}) — Ken Burns GIF")
    if backend == "svd":
        print("⚠️  SVD backend removed — use kandinsky5_i2v or kenburns")
    return kenburns_gif(image_path, out.with_suffix(".gif"), frames=16)
