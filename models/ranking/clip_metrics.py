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

    def _project_if_needed(self, pooled: torch.Tensor, *, image: bool) -> torch.Tensor:
        proj_name = "visual_projection" if image else "text_projection"
        proj = getattr(self.model, proj_name, None)
        if proj is None:
            return pooled
        in_dim = int(proj.weight.shape[1])
        dim = int(pooled.shape[-1])
        if dim == in_dim:
            x = pooled.detach().to(device=proj.weight.device, dtype=proj.weight.dtype)
            return proj(x)
        return pooled.detach()

    def _encoder_hidden(self, feat, *, image: bool):
        inner = getattr(feat, "vision_model_output" if image else "text_model_output", None)
        pooled = getattr(feat, "pooler_output", None)
        if pooled is None and inner is not None:
            pooled = getattr(inner, "pooler_output", None)
        hs = getattr(feat, "last_hidden_state", None)
        if hs is None and inner is not None:
            hs = getattr(inner, "last_hidden_state", None)
        return pooled, hs

    def _to_embed(self, feat, *, image: bool) -> torch.Tensor:
        """Map CLIP outputs into the shared 768-d (ViT-L) embedding space.

        get_*_features already returns the projected vector — do not project again.
        Encoder ModelOutput needs last_hidden (1024 for vision) then visual_projection.
        """
        if torch.is_tensor(feat):
            t = feat
        else:
            t = None
            for attr in ("image_embeds", "text_embeds"):
                val = getattr(feat, attr, None)
                if val is not None and torch.is_tensor(val):
                    t = val
                    break
            if t is None:
                pooled, hs = self._encoder_hidden(feat, image=image)
                proj = getattr(self.model, "visual_projection" if image else "text_projection", None)
                in_dim = int(proj.weight.shape[1]) if proj is not None else None
                # Prefer the pre-projection hidden state (ViT-L vision = 1024).
                if hs is not None and torch.is_tensor(hs) and in_dim and int(hs.shape[-1]) == in_dim:
                    t = self._project_if_needed(hs[:, 0], image=image)
                elif pooled is not None and torch.is_tensor(pooled) and in_dim and int(pooled.shape[-1]) == in_dim:
                    t = self._project_if_needed(pooled, image=image)
                elif pooled is not None and torch.is_tensor(pooled):
                    t = pooled
                elif hs is not None and torch.is_tensor(hs):
                    t = self._project_if_needed(hs[:, 0], image=image)
            if t is None and isinstance(feat, (tuple, list)) and feat and torch.is_tensor(feat[0]):
                t = feat[0]
        if t is None:
            raise TypeError(f"CLIP returned {type(feat).__name__}, expected a tensor")
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    @torch.inference_mode()
    def _image_embed(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        try:
            feat = self.model.get_image_features(pixel_values=pixel)
        except TypeError:
            feat = self.model.get_image_features(**inputs.to(self.device))
        feat = self._to_embed(feat, image=True)
        return F.normalize(feat.float(), dim=-1).detach().clone()

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
        return F.normalize(feat.float(), dim=-1).detach().clone()

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
        img_emb = self._image_embed(gen)
        txt_emb = self._text_embed(prompt_text or "commercial product photography")
        cos_t = float((img_emb @ txt_emb.T).squeeze().cpu())

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
