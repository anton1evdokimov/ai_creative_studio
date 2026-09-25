from pathlib import Path

import yaml

HF_MODELS = {
    "schnell": "black-forest-labs/FLUX.1-schnell",
    "dev": "black-forest-labs/FLUX.1-dev",
    "krea-dev": "black-forest-labs/FLUX.1-Krea-dev",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "turbo": "stabilityai/sdxl-turbo",
}

_CONFIG_CACHE = None


def _load_root_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(path) as file:
        _CONFIG_CACHE = yaml.safe_load(file) or {}
    return _CONFIG_CACHE


def load_diffusion_config() -> dict:
    data = _load_root_config()
    config = data.get("diffusion") or {}

    model = config.get("model", "sdxl-turbo")
    name = str(model).lower()
    is_schnell = "schnell" in name
    is_turbo = "turbo" in name or name in {"sdxl-turbo"}
    is_sdxl_base = name in {"sdxl"} or "stable-diffusion-xl-base" in name

    if is_turbo:
        default_steps, default_guidance, default_size = 4, 0.0, 512
    elif is_schnell:
        default_steps, default_guidance, default_size = 4, 0.0, 832
    elif is_sdxl_base:
        default_steps, default_guidance, default_size = 20, 5.0, 768
    else:
        default_steps, default_guidance, default_size = 28, 3.5, 832

    defaults = {
        "model": model,
        "quantize": 4,
        "width": default_size,
        "height": default_size,
        "num_inference_steps": default_steps,
        "guidance_scale": default_guidance,
        "output_dir": "generated",
        "lora": {
            "enabled": False,
            "path": "",
            "scale": 0.8,
            "trigger": "",
        },
        "controlnet": {
            "enabled": False,
            "modes": ["openpose", "depth"],
            "openpose_scale": 0.55,
            "depth_scale": 0.65,
            "openpose_model": "thibaud/controlnet-openpose-sdxl-1.0",
            "depth_model": "diffusers/controlnet-depth-sdxl-1.0",
            "save_maps": True,
            "image": "",
        },
        "ip_adapter": {
            "enabled": False,
            "scale": 0.6,
            "image": "",
        },
    }

    merged = {**defaults, **config}
    lora_cfg = defaults["lora"] | (config.get("lora") or {} if isinstance(config.get("lora"), dict) else {})
    merged["lora"] = lora_cfg
    cn_defaults = defaults["controlnet"]
    cn_src = config.get("controlnet") if isinstance(config.get("controlnet"), dict) else {}
    merged["controlnet"] = {**cn_defaults, **cn_src}
    modes = merged["controlnet"].get("modes") or cn_defaults["modes"]
    if isinstance(modes, str):
        modes = [m.strip() for m in modes.split(",") if m.strip()]
    merged["controlnet"]["modes"] = modes
    cn_image = str(merged["controlnet"].get("image") or "").strip()
    if cn_image:
        ip = Path(cn_image)
        if not ip.is_absolute():
            ip = Path(__file__).resolve().parents[2] / ip
        merged["controlnet"]["image"] = str(ip)
    ipa_defaults = defaults["ip_adapter"]
    ipa_src = config.get("ip_adapter") if isinstance(config.get("ip_adapter"), dict) else {}
    merged["ip_adapter"] = {**ipa_defaults, **ipa_src}
    ipa_image = str(merged["ip_adapter"].get("image") or "").strip()
    if ipa_image:
        ipa = Path(ipa_image)
        if not ipa.is_absolute():
            ipa = Path(__file__).resolve().parents[2] / ipa
        merged["ip_adapter"]["image"] = str(ipa)
    # Make output_dir absolute if relative
    out = Path(merged["output_dir"])
    if not out.is_absolute():
        merged["output_dir"] = str((Path(__file__).resolve().parents[2] / out).resolve())
    lora_path = str(lora_cfg.get("path") or "").strip()
    if lora_path:
        lp = Path(lora_path)
        if not lp.is_absolute():
            lp = Path(__file__).resolve().parents[2] / lp
        merged["lora"]["path"] = str(lp)
    return merged


def load_llm_config() -> dict:
    data = _load_root_config()
    config = data.get("llm") or {}
    vlm_cfg = data.get("vlm") or {}
    defaults = {
        "mlx_model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "cuda_model": "Qwen/Qwen2.5-3B-Instruct",
        "mlx_vlm_model": vlm_cfg.get("mlx_vlm_model") if isinstance(vlm_cfg, dict) else "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        "max_tokens": 800,
        "temperature": 0.7,
    }
    return {**defaults, **config}


def load_vlm_config() -> dict:
    data = _load_root_config()
    config = data.get("vlm") or {}
    defaults = {
        "mlx_vlm_model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        "cuda_model": "Qwen/Qwen2-VL-2B-Instruct",
        "max_tokens": 900,
    }
    return {**defaults, **config}


def load_ranking_config() -> dict:
    data = _load_root_config()
    config = data.get("ranking") or {}
    defaults = {
        "clip": True,
        "aesthetic": True,
        "dino": True,
        "vlm_judge": False,
        "vlm_gate": {
            "min_clip_t": 0.55,
            "min_clip_i": 0.45,
            "min_pre_score": 0.55,
            "top_k": 1,
        },
        "clip_model": "openai/clip-vit-large-patch14",
        "dino_model": "facebook/dinov2-small",
        "weights": {
            "clip_t": 0.30,
            "clip_i": 0.15,
            "dino_i": 0.20,
            "aesthetic": 0.15,
            "vlm": 0.10,
            "heuristics": 0.10,
        },
    }
    merged = {**defaults, **config}
    wsrc = config.get("weights") if isinstance(config.get("weights"), dict) else {}
    merged["weights"] = {**defaults["weights"], **wsrc}
    gsrc = config.get("vlm_gate") if isinstance(config.get("vlm_gate"), dict) else {}
    merged["vlm_gate"] = {**defaults["vlm_gate"], **gsrc}
    return merged


def load_pipeline_config() -> dict:
    data = _load_root_config()
    config = data.get("pipeline") or {}
    defaults = {
        "num_concepts": 5,
        "top_k_concepts": 2,
        "quality_threshold": 0.0,
        "max_retries": 0,
    }
    return {**defaults, **config}


def is_sdxl_model(model: str) -> bool:
    name = str(model).lower()
    return (
        "sdxl" in name
        or "turbo" == name
        or "stable-diffusion-xl" in name
    )


def resolve_hf_model(model: str) -> str:
    return HF_MODELS.get(model, model)

