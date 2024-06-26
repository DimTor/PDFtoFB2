import argparse
import errno
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import random as rnd
import string
import sys
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from join_imgs import join_2_images
from data_generator import FakeTextDataGenerator
from string_generator import (
    create_strings_from_dict,
)

from utils import load_dict, load_fonts
from open_json import json_dataset

from parser_arg import parse_arguments

def start(count: int, out_dir: dict = {'out': 'out',
                                       'train': 'train',
                                       'test': 'test',
                                       'val': 'val'}, test_s: int = 1,
          form: int = 50, skew_angle: int = 12,
          length: int = 12, rub_chance: float = 0.3):
    """
    Для запуска не через командную строку

    Args:
        count(int):
        out_dir(dict):
        test_s(int): число о 1 до 10, где 1 - размер тестовой выборки - 10% от тренировочной, 10 - 100% от тренировочной
    """
    language = 'ab'   # словарик с символами
    blur = 2   # максимальное размытие
    random_skew = 1
    random_blur = 1  # рандомное размытие от 0 до 2
    background = 4  # фон в виде цветного изображения с шумом
    distorsion = 0  # синусовая и косинусовая волна
    distorsion_orientation = 0  # дисторшн случайный
    handwritten = 0
    name_format = 0
    width = 0    # длина текста + 10 пикселей
    alignment = 1  # выравнивание по центру
    text_color = "#282828"  # черный ширифт
    orientation = 1  # горизонтальный текст
    space_width = 0  # без пробелов
    character_spacing = 0  # без пробелов
    margins = (5, 5, 5, 5)  # отступы
    fit = False
    output_mask = 0
    word_split = False
    image_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "images")
    stroke_width = 0
    stroke_fill = "#282828"
    image_mode = "RGB"
    output_bboxes = 0
    # Argument parsing
    rub_size = test_s * count
    train_size = 10 * count
    test_size = test_s * count
    val_size = test_s * count
    mode = {'train': train_size,
            'test': test_size,
            'val': val_size}
    # Create the directory if it does not exist.

    # Creating word list

    lang_dict = load_dict(
        os.path.join(os.path.dirname(__file__), "dicts", language + ".txt")
    )
    rub_dict = load_dict(
        os.path.join(os.path.dirname(__file__), "dicts", 'rub' + ".txt")
    )

    # Create font (path) list

    fonts = load_fonts(language)
    # Creating synthetic sentences (or word)

    for i in mode:
        orientation_list = np.random.choice([0, 1], size=mode[i], p=[1 - rub_chance / 2, rub_chance / 2])
        strings = create_strings_from_dict(
            length, True, mode[i], lang_dict, rub_dict, rub_chance, orientation_list
        )

        string_count = len(strings)
        p = Pool(os.cpu_count())
        for _ in tqdm(
            p.imap_unordered(
                FakeTextDataGenerator.generate_from_tuple,
                zip(
                    [i for i in range(0, string_count)],
                    strings,
                    [fonts[rnd.randrange(0, len(fonts))] for _ in range(0, string_count)],
                    [out_dir[i]] * string_count,
                    [form + 10 * i for i in range(10)] * (string_count // 10),
                    ['jpg'] * string_count,
                    [skew_angle] * string_count,
                    [random_skew] * string_count,
                    [blur] * string_count,
                    [random_blur] * string_count,
                    [background] * string_count,
                    [distorsion] * string_count,
                    [distorsion_orientation] * string_count,
                    [handwritten] * string_count,
                    [name_format] * string_count,
                    [width] * string_count,
                    [alignment] * string_count,
                    [text_color] * string_count,
                    orientation_list,
                    [space_width] * string_count,
                    [character_spacing] * string_count,
                    [margins] * string_count,
                    [fit] * string_count,
                    [output_mask] * string_count,
                    [word_split] * string_count,
                    [image_dir] * string_count,
                    [stroke_width] * string_count,
                    [stroke_fill] * string_count,
                    [image_mode] * string_count,
                    [output_bboxes] * string_count,
                ),
            ),
            total=mode[i],
        ):
            pass
        p.terminate()
        strings = [st.replace(' ', '') for st in strings]
        for symb in rub_dict:
            if symb not in lang_dict:
                strings = [st.replace(symb, '@') if symb in st else st for st in strings]
        json_dataset(strings, i, out_dir['out'])


def main():
    """
    Description: Main function
    """

    # Argument parsing
    args = parse_arguments()
    train_size = 10 * args.count
    test_size = args.test_size * args.count
    val_size = args.test_size * args.count
    rub_ch = 0.2
    mode = {'train': train_size,
            'test': test_size,
            'val': val_size}
    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    lang_dict = load_dict(
        os.path.join(os.path.dirname(__file__), "dicts", args.language + ".txt")
    )
    rub_dict = load_dict(
        os.path.join(os.path.dirname(__file__), "dicts", 'rub' + ".txt")
    )

    fonts = load_fonts(args.language)
    # Creating synthetic sentences (or word)
    for i in mode:
        orientation_list = np.random.choice([0, 1], size=mode[i], p=[1 - args.rubbish_chance / 2, args.rubbish_chance / 2])
        strings = create_strings_from_dict(
            args.length, args.random, mode[i], lang_dict, rub_dict, args.rubbish_chance, [0]*len(orientation_list), args.mesh_prob
        )
        try:
            os.makedirs(f'{args.output_dir}/textrecog_imgs/'+i)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if args.case == "upper":
            strings = [x.upper() for x in strings]
        if args.case == "lower":
            strings = [x.lower() for x in strings]

        string_count = len(strings)
        p = Pool(args.thread_count)
        for _ in tqdm(
            p.imap_unordered(
                FakeTextDataGenerator.generate_from_tuple,
                zip(
                    [i for i in range(0, string_count)],
                    strings,
                    [fonts[rnd.randrange(0, len(fonts))] for _ in range(0, string_count)],
                    [f'{args.output_dir}/textrecog_imgs/'+i] * string_count,
                    [args.format + 10 * i for i in range(10)] * (string_count // 10),
                    [args.extension] * string_count,
                    [args.skew_angle] * string_count,
                    [args.random_skew] * string_count,
                    [args.blur] * string_count,
                    [args.random_blur] * string_count,
                    [args.background] * string_count,
                    [args.distorsion] * string_count,
                    [args.distorsion_orientation] * string_count,
                    [args.handwritten] * string_count,
                    [args.name_format] * string_count,
                    [args.width] * string_count,
                    [args.alignment] * string_count,
                    [args.text_color] * string_count,
                    [0] * string_count,
                    [args.space_width] * string_count,
                    [args.character_spacing] * string_count,
                    [args.margins] * string_count,
                    [args.fit] * string_count,
                    [args.output_mask] * string_count,
                    [args.word_split] * string_count,
                    [args.image_dir] * string_count,
                    [args.stroke_width] * string_count,
                    [args.stroke_fill] * string_count,
                    [args.image_mode] * string_count,
                    [args.output_bboxes] * string_count,
                ),
            ),
            total=mode[i],
        ):
            pass
        p.terminate()
        strings = [st.replace(' ', '') for st in strings]
        for n, st in enumerate(strings):
            if any(symb in st for symb in rub_dict):
                for s in sorted(rub_dict, key=lambda x: len(x), reverse=True):
                    st = '@' if s in st else st
            strings[n] = st
        #strings = join_2_images(strings, args.output_dir + '/textrecog_imgs/' + i, int(mode[i]*args.chance_join))
        json_dataset(strings, i, args.output_dir)


if __name__ == "__main__":
    main()