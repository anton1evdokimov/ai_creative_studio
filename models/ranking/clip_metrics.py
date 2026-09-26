"""CLIP-T / CLIP-I / LAION aesthetic scores.

Uses one CLIP ViT-L/14 encoder (same space as the official aesthetic MLP).
Loaded only during evaluation, after SDXL is unloaded.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from schemas.evaluation import CLIPScores


AESTHETIC_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac+logos+ava1-l14-linearMSE.pth"
)

_clip_instance = None


def _use_mock() -> bool:
    return os.environ.get("AICS_USE_MOCK", "").lower() in {"1", "true", "yes", "on"}


def _device() -> str:
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


class AestheticMLP(nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


_CLIP_SKIP = (
    "hasselblad",
    "phase one",
    "tlci",
    "octabox",
    "strip box",
    "8k",
    "4k",
    "masterpiece",
    "awards",
    "iq4",
    "150mp",
)


def clip_alignment_text(
    prompt: str = "",
    concept=None,
    product: dict | None = None,
) -> str:
    """Short English caption for CLIP-T. ViT-L/14 dies on 400-char refine prompts."""
    pa = product or {}
    name = str(pa.get("product_name") or pa.get("product_type") or "").strip()
    bits: list[str] = []
    if name:
        bits.append(name)
    if concept is not None:
        for attr in ("scene", "lighting", "style", "mood"):
            v = str(getattr(concept, attr, None) or "").strip()
            if v:
                bits.append(v)
    if bits:
        return ("a photograph of " + ", ".join(bits))[:240]
    skip = _CLIP_SKIP
    chunks = [c.strip() for c in (prompt or "").split(",") if c.strip()]
    kept = []
    for c in chunks:
        low = c.lower()
        if any(s in low for s in skip):
            continue
        kept.append(c)
        if len(", ".join(kept)) > 200:
            break
    body = ", ".join(kept[:8]) if kept else (prompt or "a product photograph")
    return ("a photograph of " + body)[:240]


def _clip01_from_cosine(cos: float) -> float:
    """CLIPScore-style 0..1 for image–text. Saturates at cosine 0.4."""
    return round(max(0.0, min(1.0, 2.5 * max(float(cos), 0.0))), 4)


def _pair01_from_cosine(cos: float) -> float:
    """Image–image 0..1. Do not use CLIPScore 2.5× — typical cos is already 0.4–0.8."""
    return round(max(0.0, min(1.0, (float(cos) + 1.0) / 2.0)), 4)


class CLIPMetrics:
    def __init__(self, model_id: str, use_aesthetic: bool = True, device: str | None = None):
        self.model_id = model_id
        self.use_aesthetic = use_aesthetic
        self.device = device or _device()
        self.model = None
        self.processor = None
        self.aesthetic = None
        if not _use_mock():
            self._load()

    def _load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        print(f"📎 Loading CLIP ({self.model_id}) on {self.device} fp32")
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        if self.use_aesthetic:
            self.aesthetic = self._load_aesthetic(self.model.config.projection_dim)

    def _load_aesthetic(self, dim: int) -> AestheticMLP | None:
        if dim != 768:
            print(f"⚠️  Aesthetic MLP expects 768-d (ViT-L/14); got {dim}. Skipping aesthetic.")
            return None
        cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "aesthetic"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weights = cache_dir / "sac+logos+ava1-l14-linearMSE.pth"
        if not weights.exists():
            print("   Downloading LAION aesthetic predictor weights…")
            import urllib.request

            urllib.request.urlretrieve(AESTHETIC_URL, weights)
        mlp = AestheticMLP(768)
        state = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            mlp.load_state_dict(state)
        except RuntimeError:
            mlp.layers.load_state_dict(state)
        mlp.to(self.device, dtype=torch.float32)
        mlp.eval()
        print("   Aesthetic predictor ready")
        return mlp

    def _as_embed(self, feat) -> torch.Tensor:
        if torch.is_tensor(feat):
            t = feat
        else:
            t = getattr(feat, "image_embeds", None)
            if t is None:
                t = getattr(feat, "text_embeds", None)
        if t is None:
            raise TypeError(f"CLIP returned {type(feat).__name__}, expected a tensor")
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    @torch.inference_mode()
    def _image_embed(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.device)
        feat = self.model.get_image_features(pixel_values=pixel)
        return F.normalize(self._as_embed(feat).float(), dim=-1).detach().clone()

    @torch.inference_mode()
    def score(
        self,
        image_path: str | Path,
        prompt_text: str,
        product_image_path: str | Path | None = None,
    ) -> CLIPScores:
        if _use_mock() or self.model is None:
            return CLIPScores(
                clip_t=0.72,
                clip_t_raw=0.29,
                clip_i=0.68 if product_image_path else None,
                clip_i_raw=0.27 if product_image_path else None,
                aesthetic=0.62,
                aesthetic_raw=6.2,
            )

        path = Path(image_path)
        if not path.exists():
            return CLIPScores()

        gen = Image.open(path).convert("RGB")
        query = (prompt_text or "a product photograph")[:300]
        paired = self.processor(
            text=[query],
            images=gen,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        paired = {k: v.to(self.device) for k, v in paired.items()}
        out = self.model(**paired)
        img_emb = F.normalize(out.image_embeds.float(), dim=-1)
        txt_emb = F.normalize(out.text_embeds.float(), dim=-1)
        cos_t = float((img_emb * txt_emb).sum(dim=-1).squeeze().cpu())

        clip_i = clip_i_raw = None
        if product_image_path:
            ref = Path(product_image_path)
            if ref.exists():
                prod_emb = self._image_embed(Image.open(ref).convert("RGB"))
                clip_i_raw = float((img_emb @ prod_emb.T).squeeze().cpu())
                clip_i = _pair01_from_cosine(clip_i_raw)

        aesthetic_raw = 0.0
        aesthetic = 0.0
        if self.aesthetic is not None:
            raw = self.aesthetic(img_emb.float()).squeeze()
            aesthetic_raw = float(raw.cpu())
            aesthetic = max(0.0, min(1.0, aesthetic_raw / 10.0))

        return CLIPScores(
            clip_t=_clip01_from_cosine(cos_t),
            clip_t_raw=round(cos_t, 4),
            clip_i=clip_i,
            clip_i_raw=round(clip_i_raw, 4) if clip_i_raw is not None else None,
            aesthetic=round(aesthetic, 4),
            aesthetic_raw=round(aesthetic_raw, 3),
        )


def create_clip_metrics(model_id: str, use_aesthetic: bool = True) -> CLIPMetrics:
    global _clip_instance
    if _clip_instance is None:
        _clip_instance = CLIPMetrics(model_id, use_aesthetic=use_aesthetic)
    return _clip_instance


def unload_clip_metrics() -> None:
    global _clip_instance
    if _clip_instance is None:
        return
    _clip_instance.model = None
    _clip_instance.processor = None
    _clip_instance.aesthetic = None
    _clip_instance = None
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
