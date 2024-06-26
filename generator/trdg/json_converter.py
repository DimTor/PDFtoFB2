from open_json import read_json, json_for_detection, json_for_recog
import argparse
import os
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        nargs="?",
        help="Путь к json файлу из label studio",
        default="",
    )
    parser.add_argument(
        "-s",
        "--train_size",
        type=int,
        nargs="?",
        help="Размер тренировочной выборки",
        default=0.7,
    )
    parser.add_argument(
        "-t",
        "--type",
        type=int,
        nargs="?",
        help="Тип 0: для детекции, 1: для распознования",
        default=0,
    )
    parser.add_argument(
        "-u",
        "--update",
        type=int,
        nargs="?",
        help="1: перезаписывает json файлы в директории, 0: дописывает",
        default=1,
    )
    args = parser.parse_args()
    if args.update:
        if args.type:
            if os.path.exists('textrecog_test.json'):
                os.remove('textrecog_test.json')
            if os.path.exists('textrecog_train.json'):
                os.remove('textrecog_train.json')
            if os.path.exists('textrecog_val.json'):
                os.remove('textrecog_val.json')
        else:
            if os.path.exists('textdet_test.json'):
                os.remove('textdet_test.json')
            if os.path.exists('textdet_train.json'):
                os.remove('textdet_train.json')
            if os.path.exists('textdet_val.json'):
                os.remove('textdet_val.json')
    data_json = read_json(args.path)

    train_size = int(args.train_size * len(data_json))
    test_size = (len(data_json) - train_size) // 2
    val_size = len(data_json) - train_size - test_size

    index = list(range(len(data_json)))
    random.shuffle(index)

    train_index = index[:train_size]
    test_index = index[train_size:(train_size+test_size)]
    val_index = index[(train_size+test_size):]

    if args.type == 0:
        for n, data in enumerate(data_json):
            name = data['ocr'].split('-')[-1]
            width = data['bbox'][0]['original_width']
            height = data['bbox'][0]['original_height']
            instances = []
            for i in range(len(data['bbox'])):
                box = data['bbox'][i]
                text = data['transcription'][i]
                x1, y1, x2, y2 = (box['x'] * width / 100, box['y'] * height / 100, box['x'] + box['width'] * width / 100,
                                  box['y'] + box['height'] * height / 100)
                inst = {"bbox": [x1, y1, x2, y2], "bbox_label": 0, "polygon": [x1, y1, x2, y1, x2, y2, x2, y2],
                        "text": text, "ignore": 'false'}
                instances.append(inst)
            if n in train_index:
                mode = 'train'
            elif n in test_index:
                mode = 'test'
            else:
                mode = 'val'
            value = {"img_path": f"{mode}/{name}", "width": width, "height": height, "instances": instances}
            json_for_detection(value, mode, mode)
    else:
        for n, data in enumerate(data_json):
            name = data['ocr'].split('-')[-1]
            text = data['transcription'][0]
            if n in train_index:
                mode = 'train'
            elif n in test_index:
                mode = 'test'
            else:
                mode = 'val'
            value = {"instances": [{"text": text}], "img_path": f"textrecog_imgs/{mode}/{name}"}
            json_for_recog(value, mode, mode)
    print('OK')


