from run import start
import argparse
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, nargs="?", help="Выходная директория", default="out"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="Количетво изображений",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        help="Максимальное количество символов",
        default=14,
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        help="Высота изображения",
        default=50,
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        help="Количество используемых ядер",
        default=os.cpu_count(),
    )

    parser.add_argument(
        "-k",
        "--skew_angle",
        type=int,
        nargs="?",
        help="Максимальный угол наклона",
        default=12,
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=int,
        nargs="?",
        help="Максимальный радиус размытия",
        default=2,
    )
    parser.add_argument(
        "-ts",
        "--test_size",
        type=int,
        nargs="?",
        help="число о 1 до 10, где 1 - размер тестовой выборки - 10% от тренировочной, 10 - 100% от тренировочной",
        default=1,
    )
    parser.add_argument(
        "-rub",
        "--rubbish_chance",
        type=float,
        nargs="?",
        help="Максимальный радиус размытия",
        default=0.3,
    )
    return parser.parse_args()


def main():
    args = parse()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_dir = {'out': os.path.abspath(args.output_dir)}
    for mode in ['train', 'test', 'val']:
        if not os.path.exists(f'{args.output_dir}/textrecog_imgs/{mode}'):
            os.makedirs(f'{args.output_dir}/textrecog_imgs/{mode}')
        out_dir[mode] = os.path.abspath(f'{args.output_dir}/textrecog_imgs/{mode}')
    start(args.count, out_dir)


if __name__ == '__main__':
    main()
