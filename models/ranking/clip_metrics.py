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


def _clip01_from_cosine(cos: float) -> float:
    return round(max(0.0, min(1.0, 2.5 * max(float(cos), 0.0))), 4)


class CLIPMetrics:
    def __init__(self, model_id: str, use_aesthetic: bool = True):
        self.model_id = model_id
        self.use_aesthetic = use_aesthetic
        self.device = _device()
        self.dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        self.model = None
        self.processor = None
        self.aesthetic = None
        if not _use_mock():
            self._load()

    def _load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        print(f"📎 Loading CLIP ({self.model_id}) on {self.device}")
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.model.to(self.device, dtype=self.dtype)
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

    def _to_embed(self, feat, *, image: bool) -> torch.Tensor:
        if torch.is_tensor(feat):
            return feat
        for attr in ("image_embeds", "text_embeds"):
            val = getattr(feat, attr, None)
            if val is not None and torch.is_tensor(val):
                return val
        pooled = getattr(feat, "pooler_output", None)
        if pooled is not None and torch.is_tensor(pooled):
            proj = self.model.visual_projection if image else self.model.text_projection
            return proj(pooled.to(dtype=proj.weight.dtype))
        if isinstance(feat, (tuple, list)) and feat and torch.is_tensor(feat[0]):
            return feat[0]
        raise TypeError(f"CLIP returned {type(feat).__name__}, expected a tensor")

    @torch.inference_mode()
    def _image_embed(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        try:
            feat = self.model.get_image_features(pixel_values=pixel)
        except TypeError:
            feat = self.model.get_image_features(**inputs.to(self.device))
        feat = self._to_embed(feat, image=True)
        return F.normalize(feat.float(), dim=-1)

    @torch.inference_mode()
    def _text_embed(self, text: str) -> torch.Tensor:
        inputs = self.processor(
            text=[text[:300] if text else "a product photo"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        ids = inputs["input_ids"].to(self.device)
        kwargs = {"input_ids": ids}
        mask = inputs.get("attention_mask")
        if mask is not None:
            kwargs["attention_mask"] = mask.to(self.device)
        try:
            feat = self.model.get_text_features(**kwargs)
        except TypeError:
            feat = self.model.get_text_features(input_ids=ids)
        feat = self._to_embed(feat, image=False)
        return F.normalize(feat.float(), dim=-1)

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
        img_emb = self._image_embed(gen)
        txt_emb = self._text_embed(prompt_text or "commercial product photography")
        cos_t = float((img_emb @ txt_emb.T).squeeze().cpu())

        clip_i = clip_i_raw = None
        if product_image_path:
            ref = Path(product_image_path)
            if ref.exists():
                prod_emb = self._image_embed(Image.open(ref).convert("RGB"))
                clip_i_raw = float((img_emb @ prod_emb.T).squeeze().cpu())
                clip_i = _clip01_from_cosine(clip_i_raw)

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
