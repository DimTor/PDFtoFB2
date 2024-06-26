from PIL import Image, ImageFilter, ImageStat
import os
import random


def join_2_images(strings, output_dir, count):
    good_idx = []
    rub_idx = []
    for n, st in enumerate(strings):
        if all(symb != '@' for symb in st):
            if os.path.exists(output_dir + f'/img_{n+1}.jpg'):
                good_idx.append(n)
        if all(symb == '@' for symb in st):
            if os.path.exists(output_dir + f'/img_{n+1}.jpg'):
                rub_idx.append(n)
    for _ in range(count):
        gaussian_filter = ImageFilter.GaussianBlur(radius=2)
        rnd = random.random()
        tr_indx = good_idx.pop(random.randint(0, len(good_idx) - 1))
        bd_indx = random.choice(rub_idx)
        true_imgs_path = output_dir + f'/img_{tr_indx + 1}.jpg'
        true_imgs = Image.open(true_imgs_path)
        bad_imgs = Image.open(output_dir + f'/img_{bd_indx + 1}.jpg')
        if rnd <= 0.25:
            """Добавление изображения сверху"""
            new_im = Image.new('RGB',(true_imgs.size[0], true_imgs.size[1] +  bad_imgs.size[1]), (250,250,250))
            bad_imgs = bad_imgs.resize((true_imgs.size[0], bad_imgs.size[1]))
            bad_imgs = bad_imgs.filter(gaussian_filter)
            new_im.paste(bad_imgs, (0,0))
            new_im.paste(true_imgs, (0, bad_imgs.size[1]))
        elif 0.25 < rnd <= 0.5:
            """Добавление изображения снизу"""
            new_im = Image.new('RGB', (true_imgs.size[0], true_imgs.size[1] + bad_imgs.size[1]), (250, 250, 250))
            bad_imgs = bad_imgs.resize((true_imgs.size[0], bad_imgs.size[1]))
            bad_imgs = bad_imgs.filter(gaussian_filter)
            new_im.paste(true_imgs, (0, 0))
            new_im.paste(bad_imgs, (0, true_imgs.size[1]))
        elif 0.5 < rnd <= 0.75:
            """Добавление изображения слева"""
            new_im = Image.new('RGB', (true_imgs.size[0] + bad_imgs.size[0], true_imgs.size[1]), (250, 250, 250))
            bad_imgs = bad_imgs.resize((bad_imgs.size[0], true_imgs.size[1]))
            bad_imgs = bad_imgs.filter(gaussian_filter)
            new_im.paste(bad_imgs, (0, 0))
            new_im.paste(true_imgs, (bad_imgs.size[0], 0))
            strings[tr_indx] = '@' + strings[tr_indx]
        else:
            """Добавление изображения справа"""
            new_im = Image.new('RGB', (true_imgs.size[0] + bad_imgs.size[0], true_imgs.size[1]), (250, 250, 250))
            bad_imgs = bad_imgs.resize((bad_imgs.size[0], true_imgs.size[1]))
            bad_imgs = bad_imgs.filter(gaussian_filter)
            new_im.paste(true_imgs, (0, 0))
            new_im.paste(bad_imgs, (true_imgs.size[0], 0))
            strings[tr_indx] = strings[tr_indx] + '@'
        new_im.save(true_imgs_path, 'JPEG')
        return strings
