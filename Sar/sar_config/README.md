## Обучение SAR

## Расположение
Сейчас модель ждет, что в корне лежит папка out со структурой 
вышедшой из под recog_generator
- C gjvjom. --dataset можно задать свой путь к датасету

# Запуск
Осуществляется вызовом скрипта train.py
-python train.py

После запуска пройдет секунд 20 - на подсчет среднего и 
стандартного отклонения

По дефолту 
- batch_size = 32
- частота валидации - val_interval = 10
- Количество эпох = 100
- Сохранение весов каждые 5 эпох

python train.py --config configs/textrecog/sar/sar_icdar.py --work_dir my_work_dir --dataset out






