from mmengine import Config
from mmengine.runner import Runner
import time
import argparse
from mean_std import mean_std
import os
# Load the config


def go_train(config_path, work_dir, dataset):
    '''
    Функция запускает обучение модели распознования
    '''
    print(config_path)
    if os.path.exists(dataset):
        os.rename(dataset, 'out')
    cfg = Config.fromfile(config_path)
    # Specify the work dir
    cfg.work_dir = work_dir
    train_path = cfg.icdar2015_textrecog_train.data_root + cfg.icdar2015_textrecog_train.ann_file
    test_path = cfg.icdar2015_textrecog_test.data_root + cfg.icdar2015_textrecog_test.ann_file
    root = cfg.icdar2015_textrecog_test.data_root
    mean, std = mean_std('/home/tor/PycharmProjects/Samsung/Sar/recognition_generator/out/textrecog_imgs/train', cfg.train_pipeline[2])
    cfg.model.data_preprocessor.mean = mean
    cfg.model.data_preprocessor.std = std
    # Configure the batch size, learning rate, and maximum epochs
    cfg.train_dataloader.batch_size = 32
    cfg.train_cfg.val_interval = 1
    cfg.train_cfg.max_epochs = 10
    # Save checkpoint every 5 epochs
    cfg.default_hooks.checkpoint.interval = 1

    # Set seed thus the results are more reproducible
    cfg.randomness = dict(seed=0)

    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')

    # created in multiple runs
    cfg.visualizer.name = f'{time.localtime()}'

    runner = Runner.from_cfg(cfg)
    runner.train()
    os.rename('out', dataset)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-conf",
        "--config",
        type=str,
        nargs="?",
        help="Путь к файлу с config",
        default='configs/textrecog/sar/sar_icdar.py',
    )
    parser.add_argument(
        "-w",
        "--work_dir",
        type=str,
        nargs="?",
        help="Путь к папке куда сохранять веса модели",
        default='work_dir',
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        nargs="?",
        help="Путь к датасету",
        default='out',
    )
    args = parser.parse_args()
    go_train(args.config, args.work_dir, args.dataset)


if __name__ == '__main__':
    parse()
