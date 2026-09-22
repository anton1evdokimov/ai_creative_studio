import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL

# 1. Задаем устройство MPS (Metal Performance Shaders) для Apple Silicon
device = "mps"

# 2. CV-этап: Готовим опорное изображение (контуры Canny)
# Замените 'input_structure.jpg' на любую вашу картинку
try:
    image_path = "test.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError
except FileNotFoundError:
    # Если файла нет, создаем тестовый белый квадрат с черным кругом по центру
    image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    cv2.circle(image, (256, 256), 100, (0, 0, 0), -1)

# Выделяем контуры алгоритмом Canny
low_threshold = 100
high_threshold = 200
edges = cv2.Canny(image, low_threshold, high_threshold)
edges = edges[:, :, None]
edges = np.concatenate([edges, edges, edges], axis=2)
canny_image = Image.fromarray(edges)
canny_image.save("detected_maps.png") # Сохраним карту контуров для проверки

print("Карта контуров Canny успешно подготовлена.")

# 3. Загружаем веса ControlNet для обработки Canny-контуров (под SDXL)
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", 
    dtype=torch.float16
)

# 4. Собираем единый пайплайн SDXL + ControlNet
print("Загрузка базовой модели и ControlNet на Макбук...")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    dtype=torch.float16
).to(device)

# Оптимизация для Mac: очистка кэша MPS
torch.mps.empty_cache()

# 5. Промпт и параметры генерации
prompt = "a futuristic cyber-punk neon building, highly detailed, 8k resolution"
negative_prompt = "blurry, low quality, distorted"

print("Запуск диффузии по контурам...")
output = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=canny_image,          # Передаем нашу CV-карту контуров
    controlnet_conditioning_scale=0.8, # Жесткость удержания контуров (0.0 - 1.0)
    num_inference_steps=30,     # Количество шагов сэмплинга
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

# 6. Сохраняем результат
output.save("controlnet_output.png")
print("Готово! Результат сохранен в файл controlnet_output.png")
