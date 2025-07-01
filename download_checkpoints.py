import os
import torch

from diffusers import StableDiffusionImg2ImgPipeline, PNDMScheduler
from huggingface_hub import hf_hub_download

# выбор устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ID основной модели
model_id = "hafsa000/interior-design"

# список LoRA-чекипоинтов
sdxl_loras = [
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
    "xsarchitectural-19Houseplan.safetensors",
    "xsarchitectural-7.safetensors"
]


def fetch_checkpoints():
    """
    Скачивает IP-Adapter + все LoRA-чекипоинты в папку ./loras
    """
    # создаём папку для loras, если её нет
    os.makedirs("./loras", exist_ok=True)

    # IP-Adapter
    hf_hub_download(
        repo_id='h94/IP-Adapter-FaceID',
        filename='ip-adapter-faceid-portrait_sdxl.bin',
        local_dir='./',
        local_dir_use_symlinks=False
    )

    # все LoRA
    for weights in sdxl_loras:
        print(f"Downloading {weights} …")
        hf_hub_download(
            repo_id="sintecs/interior",
            filename=weights,
            local_dir="./loras",
            local_dir_use_symlinks=False,
            force_download=True,
            resume_download=False
        )


def fetch_pretrained_model(model_id=model_id, **kwargs):
    """
    Загружает Img2Img pipeline и заменяет планировщик на PNDM
    """
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        **kwargs
    )
    # заменяем scheduler, чтобы он был PNDM
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    # переносим на GPU/CPU
    pipe = pipe.to(device)
    return pipe


if __name__ == '__main__':
    # 1) скачиваем чекпоинты
    fetch_checkpoints()

    # 2) инициализируем pipeline
    pipe = fetch_pretrained_model()
