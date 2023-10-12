from PIL import Image
import numpy as np
import cv2
import time


def process_image(image_path, threshold, erosion_step):
    # Загрузка изображения
    img = Image.open(image_path)
    img_array = np.array(img)

    # Вычисление интенсивности
    intensity = np.sum(img_array, axis=2) / 3

    # Применение порога
    binary_image = np.where(intensity < threshold, 0, 1)

    # Эрозия
    kernel = np.ones((erosion_step, erosion_step), np.uint8)
    eroded_image = cv2.erode(binary_image.astype(np.uint8), kernel, iterations=1)

    # Преобразование результатов в изображение
    result_image = Image.fromarray((eroded_image * 255).astype(np.uint8))

    return result_image


def main():
    image_sizes = [(1024, 768), (1280, 960), (2048, 1536)]
    avg_time = 0  # Среднее время работы программы
    threshold = 150  # Порог интенсивности
    erosion_step = 2  # Шаг эрозии

    for size in image_sizes:
        for _ in range(3):  # Троекратный перезапуск
            start = time.time()

            image_path = f"hl_image_{size[0]}x{size[1]}.jpg"
            result_image = process_image(image_path, threshold, erosion_step)

            # Сохранение результата
            result_image.save(f"result_{size[0]}x{size[1]}_t{threshold}_e{erosion_step}.png")

            end = time.time() - start
            avg_time += end

        print(f"Среднее время работы программы для файлов {size[0]}x{size[1]}: "
              f"{float('{:.5f}'.format(avg_time / 9))} сек.")
        avg_time = 0


if __name__ == "__main__":
    main()
