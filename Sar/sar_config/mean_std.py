from PIL import Image
import numpy as np
import os
from mmocr.datasets.transforms.textrecog_transforms import RescaleToHeight


def mean_std(root: str, preprocessing: dict) -> tuple:
    '''
    Функция находит среднее и стандартное отклонение пикселей в каждом канале
    Args:
        root(str): путь к папке с изображениями
        preprocessing(dict): словарь с параметрами RescaleToHeight применяемая моделью

    Returns:
        Кортеж с массивами значений среднего и стандартного отклонения
    '''
    if not os.path.exists(root):
        print(f'По пути {root} ничего не найдено, функция вернула стандартные значения')
        return ((127, 127, 127),
                (80, 80, 80))
    red_array, green_array, blue_array = [], [], []
    for filename in os.listdir(root):
        f = os.path.join(root, filename)
        # checking if it is a file
        try:
            img = Image.open(f)
            s = np.asarray(img)
            res = RescaleToHeight(height=preprocessing['height'], min_width=preprocessing['min_width'],
                                  max_width=preprocessing['max_width'],
                                  width_divisor=preprocessing['width_divisor'])
            img = Image.fromarray(np.uint8(res.transform(results={'img': s})['img']))
            pix = img.load()
            weight, height = img.size
            for x in range(weight):
                for y in range(height):
                    r, g, b = pix[x, y]
                    red_array.append(r)
                    green_array.append(g)
                    blue_array.append(b)
        except IOError:
            print('No image file')
    return ([int(np.mean(red_array)), int(np.mean(green_array)), int(np.mean(blue_array))],
            [int(np.std(red_array)), int(np.std(green_array)), int(np.std(blue_array))])


if __name__ == '__main__':
    di = dict(
                height=32,
                max_width=128,
                min_width=32,
                type='RescaleToHeight',
                width_divisor=4)
    print(mean_std('out/textrecog_imgs/train', di))
