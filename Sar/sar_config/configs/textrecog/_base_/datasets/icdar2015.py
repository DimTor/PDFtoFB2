icdar2015_textrecog_data_root = '../recognition_generator/out/'

icdar2015_textrecog_train = dict(
    type='OCRDataset',
    data_root=icdar2015_textrecog_data_root,
    ann_file='train_json.json',
    pipeline=None)

icdar2015_textrecog_test = dict(
    type='OCRDataset',
    data_root=icdar2015_textrecog_data_root,
    ann_file='val_json.json',
    test_mode=True,
    pipeline=None)

icdar2015_1811_textrecog_test = dict(
    type='OCRDataset',
    data_root=icdar2015_textrecog_data_root,
    ann_file='textrecog_test_1811.json',
    test_mode=True,
    pipeline=None)
