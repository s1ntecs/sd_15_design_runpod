"""RunPod handler for txt2img / img2img with optional LoRA support.

• Если «lora» не передаётся → генерируем без LoRA (дефолт).  
• Если «lora» указан → подгружаем соответствующий .safetensors-файл,
  выгружаем предыдущий, фьюзим новый, генерируем.

Запрос (job['input'])
---------------------
{
    "prompt": "cozy living room",
    "image_url": "https://…/photo.jpg",    # optional
    "lora": "XSArchi_137.safetensors",     # optional
    "strength": 0.7,
    "guidance_scale": 8,
    "steps": 22,
    "seed": 42
}
"""
from __future__ import annotations

import base64
import io
import os
import time
import traceback
from typing import Any, Dict, Optional

import runpod
import torch
from diffusers import (
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from huggingface_hub import hf_hub_download
from PIL import Image
from runpod.serverless.utils.rp_download import file as rp_file

# --------------------------------------------------------------------------- #
#                               CONFIGURATION                                 #
# --------------------------------------------------------------------------- #
MODEL_ID = "hafsa000/interior-design"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_STEPS = 125
DEFAULT_SEED = 42
LORA_REPO = "sintecs/interior"
LORA_DIR = "./loras"
LORA_LIST = [
    "XSArchi_110plan彩总.safetensors",
    "XSArchi_137.safetensors",
    "XSArchi_141.safetensors",
    "XSArchi_162BIESHU.safetensors",
    "XSarchitectural-38InteriorForBedroom.safetensors",
    "XSarchitectural_33WoodenluxurystyleV2.safetensors",
    "house_architecture_Exterior_SDlife_Chiasedamme.safetensors",
    "xsarchitectural-15Nightatmospherearchitecture.safetensors",
    "xsarchitectural-18Whiteexquisiteinterior.safetensors",
    "xsarchitectural-19Houseplan (1).safetensors",
    "xsarchitectural-19Houseplan.safetensors"
]

os.makedirs(LORA_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
#                           PIPELINE INITIALISATION                           #
# --------------------------------------------------------------------------- #
print(f"⏳ Loading pipelines on {DEVICE} …")
TXT2IMG_PIPE: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,
    requires_safety_checker=False,
).to(DEVICE)

IMG2IMG_PIPE: StableDiffusionImg2ImgPipeline = (
    StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE)
)
IMG2IMG_PIPE.scheduler = PNDMScheduler.from_config(IMG2IMG_PIPE.scheduler.config)

if DEVICE == "cuda":
    for pl in (TXT2IMG_PIPE, IMG2IMG_PIPE):
        pl.enable_attention_slicing()
        try:
            pl.enable_xformers_memory_efficient_attention()
        except Exception:  # noqa: BLE001
            pass

CURRENT_LORA: str = "None"
print("✅ Pipelines ready.")


# --------------------------------------------------------------------------- #
#                          LOADING / UNLOADING  LORA                          #
# --------------------------------------------------------------------------- #
def _download_lora(lora_name: str) -> str:
    """Ensure LoRA file is present locally and return its path."""
    local_path = os.path.join(LORA_DIR, lora_name)
    if not os.path.exists(local_path):
        hf_hub_download(
            repo_id=LORA_REPO,
            filename=lora_name,
            local_dir=LORA_DIR,
            local_dir_use_symlinks=False,
            force_download=False,
            resume_download=True,
        )
    return local_path


def _switch_lora(lora_name: Optional[str]) -> Optional[str]:
    """Load new LoRA or unload if lora_name is None. Return error str or None."""
    global CURRENT_LORA

    # --------- unload current LoRA --------- #
    if lora_name is None and CURRENT_LORA != "None":
        for pl in (TXT2IMG_PIPE, IMG2IMG_PIPE):
            pl.unfuse_lora()
            pl.unload_lora_weights()
        CURRENT_LORA = "None"
        return None

    # ----- nothing to do / unsupported ----- #
    if lora_name is None or lora_name == CURRENT_LORA:
        return None
    if lora_name not in LORA_LIST:
        return f"Unknown lora '{lora_name}'."

    # --------- load new LoRA --------------- #
    try:
        path = _download_lora(lora_name)
        # remove previous lora if any
        if CURRENT_LORA != "None":
            for pl in (TXT2IMG_PIPE, IMG2IMG_PIPE):
                pl.unfuse_lora()
                pl.unload_lora_weights()

        for pl in (TXT2IMG_PIPE, IMG2IMG_PIPE):
            pl.load_lora_weights(path)
            pl.fuse_lora()

        CURRENT_LORA = lora_name
        return None
    except Exception as err:  # noqa: BLE001
        return f"Failed to load LoRA '{lora_name}': {err}"


# --------------------------------------------------------------------------- #
#                              UTILITY FUNCS                                  #
# --------------------------------------------------------------------------- #
def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------- #
#                                 HANDLER                                     #
# --------------------------------------------------------------------------- #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler."""
    try:
        payload: Dict[str, Any] = job.get("input", {})
        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        # ------------------- handle LoRA ------------------- #
        lora_name = payload.get("lora")
        error = _switch_lora(lora_name)
        if error:
            return {"error": error}

        # ------------------- parameters -------------------- #
        image_url = payload.get("image_url")
        strength = float(payload.get("strength", 0.7))
        guidance_scale = float(payload.get("guidance_scale", 7.5))
        steps = min(int(payload.get("steps", MAX_STEPS)), MAX_STEPS)
        seed = int(payload.get("seed", DEFAULT_SEED))
        height = int(payload.get("height", 768))
        width = int(payload.get("width", 1024))
        num_images = int(payload.get("num_images", 1))
        if num_images < 1 or num_images > 8:
            return {"error": "'num_images' must be between 1 and 8."}
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        start = time.time()

        # ------------------- generation -------------------- #
        if image_url:
            init_img = url_to_pil(image_url).resize((512, 512))
            images = IMG2IMG_PIPE(
                prompt=prompt,
                image=init_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
                num_images_per_prompt=num_images,
            ).images
        else:
            images = TXT2IMG_PIPE(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
                height=height,
                width=width,
                num_images_per_prompt=num_images,
            ).images

        elapsed = round(time.time() - start, 2)
        return {
            "images_base64": [pil_to_b64(img) for img in images],
            "time": elapsed,
            "steps": steps,
            "seed": seed,
        }

    except torch.cuda.OutOfMemoryError:
        return {"error": "CUDA out of memory — reduce 'steps' or image size."}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "trace": traceback.format_exc(limit=1)}


# --------------------------------------------------------------------------- #
#                                RUN WORKER                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})