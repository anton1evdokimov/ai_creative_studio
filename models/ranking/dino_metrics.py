"""DINOv2 image–image similarity (identity / appearance, not prompt).

Complements CLIP-I: DINO is trained self-supervised on crops, so it tracks
shape/texture of the garment more than CLIP's text-aligned space.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


_dino_instance = None


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        try:
            if mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


class DinoMetrics:
    def __init__(self, model_id: str = "facebook/dinov2-small", device: str | None = None):
        self.model_id = model_id
        self.device = device or _device()
        self.model = None
        self.processor = None
        self._load()

    def _load(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        print(f"🦕 Loading DINOv2 ({self.model_id}) on {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        if getattr(out, "pooler_output", None) is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0]
        return F.normalize(feat.float(), dim=-1).detach().clone()

    def similarity(self, gen_path: str | Path, ref_path: str | Path) -> tuple[float, float]:
        """Returns (score_0_1, raw_cosine)."""
        gen = Image.open(gen_path).convert("RGB")
        ref = Image.open(ref_path).convert("RGB")
        a = self.embed(gen)
        b = self.embed(ref)
        cos = float((a @ b.T).squeeze().cpu())
        score = round(max(0.0, min(1.0, (cos + 1.0) / 2.0)), 4)
        return score, round(cos, 4)

    def similarity_mean(self, gen_path: str | Path, ref_paths: list[Path]) -> tuple[float | None, float | None]:
        existing = [p for p in ref_paths if Path(p).exists()]
        if not existing:
            return None, None
        scores, raws = [], []
        for p in existing:
            s, r = self.similarity(gen_path, p)
            scores.append(s)
            raws.append(r)
        return round(sum(scores) / len(scores), 4), round(sum(raws) / len(raws), 4)


def create_dino_metrics(model_id: str = "facebook/dinov2-small", device: str | None = None) -> DinoMetrics:
    global _dino_instance
    if _dino_instance is None:
        _dino_instance = DinoMetrics(model_id, device=device)
    return _dino_instance


def unload_dino_metrics() -> None:
    global _dino_instance
    if _dino_instance is None:
        return
    _dino_instance.model = None
    _dino_instance.processor = None
    _dino_instance = None
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
