from pathlib import Path
import random
import time

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .base import ImageBackend


_PALETTES = [
    # Minimalist Studio: whites, beiges
    ((250, 248, 244), (230, 220, 205), (180, 160, 140), (255, 255, 255)),
    # Golden Botanical: warm greens, golds
    ((247, 231, 206), (139, 154, 107), (212, 165, 116), (250, 240, 220)),
    # Deep Blue Night: navy, burgundy
    ((26, 33, 62), (15, 52, 96), (233, 69, 96), (20, 20, 40)),
    # Morning Vanity: warm wood, linen
    ((250, 243, 224), (234, 191, 159), (182, 137, 115), (255, 250, 240)),
    # Water Splash: aqua, white, transparent
    ((220, 240, 250), (100, 170, 210), (40, 90, 140), (255, 255, 255)),
]


class MockImageBackend(ImageBackend):
    """Generates pretty concept-card placeholder images with PIL — no ML model needed.

    Produces visually distinct images for each concept so you can test the full
    pipeline (save paths, naming, evaluation, etc.) without downloading FLUX.
    """

    def __init__(self, config: dict, latency_seconds: float = 0.8):
        self.config = config
        self.latency = latency_seconds
        self._calls = 0
        print("🧪 Using MockImageBackend (fast offline testing mode — no real diffusion)")

    def generate(
        self,
        prompt: str,
        output_path: str,
        seed: int | None = None,
        negative_prompt: str | None = None,
        **extra,
    ) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        time.sleep(self.latency)

        width = int(self.config.get("width", 832))
        height = int(self.config.get("height", 832))
        rng = random.Random(seed if seed is not None else self._calls)
        self._calls += 1

        palette_idx = rng.randrange(len(_PALETTES))
        bg, accent1, accent2, fg = _PALETTES[palette_idx]

        # --- Build background gradient ---
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)
        for y in range(height):
            t = y / max(1, height - 1)
            r = int(bg[0] * (1 - t) + accent1[0] * t * 0.35)
            g = int(bg[1] * (1 - t) + accent1[1] * t * 0.35)
            b = int(bg[2] * (1 - t) + accent1[2] * t * 0.35)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # --- Soft bokeh circles ---
        for _ in range(28):
            cx = rng.randint(0, width)
            cy = rng.randint(0, height)
            cr = rng.randint(width // 20, width // 5)
            alpha_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ad = ImageDraw.Draw(alpha_layer)
            col = (accent1[0], accent1[1], accent1[2], 28)
            ad.ellipse((cx - cr, cy - cr, cx + cr, cy + cr), fill=col)
            img = Image.alpha_composite(img.convert("RGBA"), alpha_layer).convert("RGB")

        # --- Product shape (rounded rectangle bottle placeholder) ---
        bw, bh = width // 5, height // 3
        bx = (width - bw) // 2
        by = (height - bh) // 2 + height // 16

        # Bottle shadow
        shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.rounded_rectangle(
            (bx + 6, by + bh - 6, bx + bw + 6, by + bh + 18),
            radius=30,
            fill=(0, 0, 0, 70),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))
        img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")

        # Bottle glass body with gradient
        bottle = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bottle)
        for yy in range(by, by + bh):
            tt = (yy - by) / max(1, bh - 1)
            rr = int(accent2[0] * 0.55 + 200 * (1 - tt))
            gg = int(accent2[1] * 0.55 + 220 * (1 - tt))
            bb = int(accent2[2] * 0.55 + 235 * (1 - tt))
            bd.line([(bx, yy), (bx + bw, yy)], fill=(rr, gg, bb, 210))
        bd.rounded_rectangle((bx, by, bx + bw, by + bh), radius=22, outline=(fg[0], fg[1], fg[2], 220), width=2)

        # Cap
        cap_h = bh // 5
        cap_w = int(bw * 0.6)
        cap_x = bx + (bw - cap_w) // 2
        cap_y = by - cap_h
        bd.rounded_rectangle(
            (cap_x, cap_y, cap_x + cap_w, by + 6),
            radius=10,
            fill=(accent1[0], accent1[1], accent1[2], 240),
            outline=(fg[0], fg[1], fg[2], 220),
            width=2,
        )

        # Highlight streak
        for yy in range(by + 8, by + bh - 12, 2):
            bd.line(
                [(bx + bw // 6, yy), (bx + bw // 6 + 4, yy)],
                fill=(255, 255, 255, 160),
            )

        img = Image.alpha_composite(img.convert("RGBA"), bottle).convert("RGB")

        # --- Title + label bar ---
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
            small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Bottom concept name banner (pasted at bottom of full-size overlay)
        banner_h = 96
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([(0, height - banner_h), (width, height)], fill=(0, 0, 0, 150))
        img = img.convert("RGBA") if img.mode != "RGBA" else img
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        # Derive a short concept label from the prompt (first 60 chars after "Scene:" or fallback)
        label = "Concept"
        for keyword in ["Scene:", "Style:", "Professional advertising photography of"]:
            if keyword in prompt:
                idx = prompt.index(keyword) + len(keyword)
                label = prompt[idx : idx + 80].strip().split(".")[0].strip()
                break
        if len(label) > 64:
            label = label[:61] + "..."

        # Seed/idx on top-right
        draw.text((width - 96, 14), f"#{seed if seed is not None else 0}", fill=(255, 255, 255, 200), font=small_font)

        draw.text((28, height - banner_h + 14), label, fill=(255, 255, 255, 240), font=font)
        draw.text((28, height - banner_h + 48), f"{width}x{height}  Mock FLUX.1 Output", fill=(220, 220, 220, 180), font=small_font)

        # Corner frame
        draw.rounded_rectangle((10, 10, width - 10, height - 10), radius=22, outline=(255, 255, 255, 80), width=1)

        img = img.convert("RGB")
        img.save(output_path, "PNG", optimize=True)
        return output_path
