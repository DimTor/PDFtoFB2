from PIL import Image, ImageDraw
import os
import errno
import random
import cv2
import numpy as np
import tqdm

def box(ann):
    with open(ann, 'r') as file:
        data = file.read().split('\n')[:-1]
        lines = []
        mass_with_word = []
        for i in data:
            try:
                cords = [int(j) for j in i.split()]
                mass_with_word.append(cords)
            except:
                x1, y1 = min(mass_with_word, key=lambda x: x[0])[0], min(mass_with_word, key=lambda x: x[1])[1]

                x2, y2 = max(mass_with_word, key=lambda x: x[2])[2], max(mass_with_word, key=lambda x: x[3])[3]
                mass_with_word = []
                lines.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
        if mass_with_word:
            x1, y1 = min(mass_with_word, key=lambda x: x[0])[0], min(mass_with_word, key=lambda x: x[1])[1]
            x2, y2 = max(mass_with_word, key=lambda x: x[2])[2], max(mass_with_word, key=lambda x: x[3])[3]
            lines.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
        return lines


def formula_to_box(image, ann):
    weight, height = Image.open(image).size
    with open(ann, 'r') as an:
        data = an.read().split('\n')[:-1]
    mass_with_word = []
    for i in data:
        try:
            cords = [float(j) for j in i.split()]
            mass_with_word.append(cords)
        except:
            pass
    try:
        x1, y1 = min(mass_with_word, key=lambda x: x[0])[0], min(mass_with_word, key=lambda x: x[1])[1]
        x2, y2 = max(mass_with_word, key=lambda x: x[2])[2], max(mass_with_word, key=lambda x: x[3])[3]
        nw = [1, (x1 + (x2 - x1) / 2) / weight, (y1 + (y2 - y1) / 2) / height, (x2 - x1) / weight, (y2 - y1) / height]
    except:
        print(mass_with_word, ann)
    with open(ann, 'w') as new_ann:
        try:
            mas = [str(d) for d in nw]
        except Exception:
            mas = ['1 0.5 0.5 0.99 0.99']
        new_ann.write(' '.join(mas) + '\n')

def create_bbox_word(image, ann):
    st = []
    weight, height = Image.open(image).size
    with open(ann, 'r') as file:
        data = file.read().split('\n')[:-1]
        mass_with_word = []
        for i in data:
            try:
                cords = [float(j) for j in i.split()]
                mass_with_word.append(cords)
            except:
                x1, y1 = min(mass_with_word, key=lambda x: x[0])[0], min(mass_with_word, key=lambda x: x[1])[1]

                x2, y2 = max(mass_with_word, key=lambda x: x[2])[2], max(mass_with_word, key=lambda x: x[3])[3]
                mass_with_word = []
                st.append([0, (x1 + (x2 - x1) / 2) / weight, (y1 + (y2 - y1) / 2) / height, (x2 - x1) / weight, (y2 - y1) / height])
        if mass_with_word:
            x1, y1 = min(mass_with_word, key=lambda x: x[0])[0], min(mass_with_word, key=lambda x: x[1])[1]
            x2, y2 = max(mass_with_word, key=lambda x: x[2])[2], max(mass_with_word, key=lambda x: x[3])[3]
            st.append([0, (x1 + (x2 - x1) / 2) / weight, (y1 + (y2 - y1) / 2) / height, (x2 - x1) / weight, (y2 - y1) / height])
        name = image.split('/')[-1].split('.')[0]
    with open(ann, 'w') as an_file:
        for s in st:
            mas = [str(d) for d in s]
            an_file.write(' '.join(mas) + '\n')


def show_bbox(path, ann):
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    lines = box(ann)
    for z in lines:
        draw.line(
            xy=z, fill='red', width=2)
    image.show()

def yolo_box(ann, im_w, im_h):
    with open(ann, 'r') as file:
        data = file.read().split('\n')[:-1]
        lines = []
        for i in data:
            lab, x, y, width, height = coords = [float(j) for j in i.split()]
            x1 = (x - width / 2) * im_w
            y1 = (y - height / 2) * im_h
            x2 = (x + width / 2) * im_w
            y2 = (y + height / 2) * im_h
            lines.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
    return lines


def show_yolo_box(path, ann):
    image = Image.open(path)
    weight, height = image.size
    draw = ImageDraw.Draw(image)
    lines = yolo_box(ann, weight, height)
    for z in lines:
        draw.line(
            xy=z, fill='red', width=2)
    image.show()


def ret_float_coords(ann):
    with open(ann, 'r') as ann_file:
        annotation = []
        data = ann_file.read().split('\n')[:-1]
        for i in data:
            annotation.append([float(j) for j in i.split()])
        return annotation


def join_imgs(list_imgs, ann_list, col_n, count, out_im, out_ann, chance, form_imgs, form_ann):
    for i in range(count):
        annotation = []
        h = 0
        w = 0
        m_w = 0
        w_f = 0
        for j in form_imgs:
            image = Image.open(j)
            im_w, im_h = image.size
            if im_w > w_f:
                w_f = im_w
        r_av, g_av, b_av = 0, 0, 0
        indx_list = random.choices(list(range(len(list_imgs))), k=col_n)
        for n, idx in enumerate(indx_list):
            image = Image.open(out_im+'/'+list_imgs[idx])
            im_w, im_h = image.size
            h += im_h
            rr, gg, bb = 0, 0, 0
            pix = 0
            """            for x in range(im_w):
                            for y in range(im_h):
                                r, g, b = image.getpixel((x, y))
                                if r + g + b >= 10:
                                    rr += r
                                    gg += g
                                    bb += b
                                    pix += 1
                        r_av += rr // pix
                        g_av += gg // pix
                        b_av += bb // pix"""
            if im_w > w:
                w = im_w
                m_w = im_w
        w += w_f // 2
        image = np.ones((h, w)) * 255
        # We add gaussian noise
        cv2.randn(image, 235, 10)
        new_im = Image.fromarray(image).convert("RGB")
        # new_im = Image.new('RGB', (w, h), (r_av // 3, g_av // 3, b_av // 3))
        now_h = 0
        max_w = 0
        for n, idx in enumerate(indx_list):
            image = Image.open(out_im+'/'+list_imgs[idx])
           # show_yolo_box(out_im+'/'+list_imgs[idx], out_ann+'/'+ann_list[idx])
            im_w, im_h = image.size
            rand = random.random()
            if rand < chance:
                idx_form = random.randint(0, len(form_ann) - 1)
                form_im = Image.open(form_imgs[idx_form])
                w_f, h_f = form_im.size
                form_im = form_im.resize((w_f, im_h))
                new_w_f, new_h_f = form_im.size
                if new_w_f > max_w:
                    max_w = new_w_f
                form_cord = ret_float_coords(form_ann[idx_form])[0]
                if random.choice([True, False]):
                    new_im.paste(image, (0, now_h))
                    new_im.paste(form_im, (im_w + 5, now_h))
                    cord = ret_float_coords(out_ann + '/' + ann_list[idx])
                    for j in cord:
                        new_x = (j[1] * im_w) / w
                        new_y = (j[2] * im_h + now_h) / h
                        new_width = j[3] * im_w / w
                        new_height = j[4] * im_h / h
                        n_c = [int(j[0]), new_x, new_y + 2 / h, new_width, new_height + 4 / h]
                        annotation.append(n_c)
                    new_x = (form_cord[1] * w_f + im_w + 5) / w
                    new_y = (form_cord[2] * new_h_f + now_h) / h
                    new_width = form_cord[3] * w_f / w
                    new_height = new_h_f / h
                    n_c = [int(form_cord[0]), new_x, new_y, new_width, new_height]
                    annotation.append(n_c)
                else:
                    new_im.paste(image, (w_f+5, now_h))
                    new_im.paste(form_im, (0, now_h))
                    cord = ret_float_coords(out_ann + '/' + ann_list[idx])
                    for j in cord:
                        new_x = (j[1] * im_w + w_f + 5) / w
                        new_y = (j[2] * im_h + now_h) / h
                        new_width = j[3] * im_w / w
                        new_height = j[4] * im_h / h
                        n_c = [int(j[0]), new_x, new_y + 2 / h, new_width, new_height + 4 / h]
                        annotation.append(n_c)
                    new_x = (form_cord[1] * w_f) / w
                    new_y = (form_cord[2] * new_h_f + now_h) / h
                    new_width = form_cord[3] * w_f / w
                    new_height = new_h_f / h
                    n_c = [int(form_cord[0]), new_x, new_y, new_width, new_height]
                    annotation.append(n_c)
            else:
                new_im.paste(image, (0, now_h))
                cord = ret_float_coords(out_ann+'/'+ann_list[idx])
                for j in cord:
                    new_x = (j[1] * im_w) / w
                    new_y = (j[2] * im_h + now_h) / h
                    new_width = j[3] * im_w / w
                    new_height = j[4] * im_h / h
                    n_c = [int(j[0]), new_x, new_y + 2/h, new_width, new_height+4/h]
                    annotation.append(n_c)
            now_h += im_h
        #new_im = new_im.crop((0, 0, m_w + 6 + max_w, h))
        with open(f'{out_ann}/{col_n}img_{i}.txt', 'w') as an:
            for st in annotation:
                mas = [str(d) for d in st]
                an.write(' '.join(mas) + '\n')
        new_im.save(f'{out_im}/{col_n}img_{i}.jpg')
     #   show_yolo_box(f'{out_im}/{col_n}img_{i}.jpg', f'{out_ann}/{col_n}img_{i}.txt')
     #   print('jjj')


def create_dataset_after_recognition(form_chance=0.9, inp='/home/tor/PycharmProjects/Samsung/generator/trdg/out', inp_form='/home/tor/PycharmProjects/Samsung/generator/trdg/out_form'):
    for i in ['test', 'train', 'val']:
        list_img = os.listdir(inp + '/images/' + i)
        list_form_img = os.listdir(inp_form + '/images/' + i)
        form_list = []
        list_form_ann = []
        list_ann = []
        for img in list_img:
            create_bbox_word(inp+'/images/'+i + '/' + img, inp+'/labels/'+i + '/' + img.split('.')[0] + '.txt')
            list_ann.append(img.split('.')[0] + '.txt')
        for img in list_form_img:
            formula_to_box(inp_form+'/images/'+i + '/' + img, inp_form+'/labels/'+i + '/' + img.split('.')[0] + '.txt')
            form_list.append(inp_form+'/images/'+i + '/' + img)
            list_form_ann.append(inp_form+'/labels/'+i + '/' + img.split('.')[0] + '.txt')
        images = os.listdir(inp + '/images/' + i)
        k = len(images)
        for j in range(9):
            join_imgs(list_img, list_ann, j+1, k // 2, inp + '/images/' + i, inp + '/labels/' + i,
                      form_chance, form_list, list_form_ann)




"""for i in range(10):
    create_bbox_word(f'test/img_{i}.jpg', f'test/{i}_boxes.txt')

for i in range(10):
    show_yolo_box(f'test/img_{i}.jpg', f'img_{i}.txt')"""


create_dataset_after_recognition()
for i in range(5):
    try:
        #formula_to_box(f'/home/tor/PycharmProjects/Samsung/generator/trdg/out_form/images/test/img_{i}.jpg', f'/home/tor/PycharmProjects/Samsung/generator/trdg/out_form/labels/test/img_{i}.txt')
        show_yolo_box(f'/home/tor/PycharmProjects/Samsung/generator/trdg/out/images/train/3img_{i}.jpg', f'/home/tor/PycharmProjects/Samsung/generator/trdg/out/labels/train/3img_{i}.txt')
    except:
        pass
lis_im = ['trdg/out_form/images/test/img_1.jpg', 'trdg/out/images/test/img_2.jpg', 'trdg/out/images/test/img_3.jpg']
ann = ['trdg/out/labels/test/img_1.txt', 'trdg/out/labels/test/img_2.txt', 'trdg/out/labels/test/img_3.txt']
#create_dataset_after_recognition()
#join_imgs(lis_im, ann, 3, 1, 'trdg/out/images/test', 'trdg/out/labels/test')
#show_yolo_box('trdg/out/images/test/img_2.jpg', 'trdg/out/labels/test/img_2.txt')