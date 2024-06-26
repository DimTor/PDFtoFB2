import json
import os
from PIL import Image


def read_json(json_path: str) -> dict:
    """
    Функция для чтения json файла

    :param json_path: str - путь к json файлу
    :return: dict - открытый json файл
    """
    with open(json_path) as file:
        my_json = json.load(file)
        return my_json


def save_json(path: str, data: dict) -> None:
    """
    Функция сохраняет json, который мжет содержать кириллицу.

    :param path: путь, по которму необходимо создать json файл
    :param data: dict, содержимое, которое необходимо записать в файл
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def json_dataset(strings: list, mode: str, output_dir: str):
    my_json = {"metainfo": {"dataset_type": "TextRecogDataset", "task_name": "textrecog"}, "data_list": []}
    for n, i in enumerate(strings):
        text = {"text": i}
        img_path = f"textrecog_imgs/{mode}/img_{n + 1}.jpg"
        if os.path.exists(output_dir + '/' + img_path):
            img_path = f"textrecog_imgs/{mode}/img_{n + 1}.jpg"
            dictanary = {"instances": [text], "img_path": img_path}
            my_json["data_list"].append(dictanary)
    save_json(f"{output_dir}/{mode}_json.json", my_json)




