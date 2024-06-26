import string

from models.Yolo_main_class import YOLOClass
import pytesseract
from models.BERT_class import BERTClass
from PIL import Image
import string
from create_fb2 import FB2
import config
from dataloader import Loader

class MainClass:
    def __init__(self):
        self.YOLOMain = YOLOClass(config.yolo_weight)
       # self.YOLOBBox = YOLOBoxClass(yolo_box_weight)
        self.BERT = BERTClass(config.bert_fold)
        self.count = 21
        self.string = ''
        self.FB2 = FB2()

    def spell_check(self, word, sentense):
        if len(word) < 4 or not all(s.lower() in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for s in word) or sentense.count(word) > 1:
            return word
        else:
            word = self.BERT.predict(sentense, [word])
            return word

    def find_word(self, box):
        pass

    def forward(self, page):
        blocks = self.YOLOMain.forward(page)
        for res in blocks:
            im = Image.open(page)
            boxes = res.boxes.cpu()  # Boxes object for bounding box outputs
            class_name = boxes.cls.numpy()
            for n, cls in enumerate(class_name):
                if cls != 1:
                    x, y, w, h = boxes.xywh.numpy()[n]
                    im_crop = im.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                    im_crop.save('out.jpg')
                    data_word = pytesseract.image_to_data('out.jpg', lang='rus+eng', output_type=pytesseract.Output.DICT)
                    new_mass_word = []
                    new_mass_word2 = []

                    for number, j in enumerate(data_word['text']):
                        punkt = ''
                        if not j:
                            continue
                        if j[-1] == '-':
                            self.string += j[:-1]
                        elif j:
                            self.string += j
                            self.string += punkt
                            new_mass_word.append(self.string)
                            self.string = ''
                    max_len = len(new_mass_word)
                    number = -1
                    for num, j in enumerate(data_word['text']):
                        punkt = ''
                        if not j:
                            continue
                        if j[-1] == '-':
                            self.string += j[:-1]
                        elif j:
                            number += 1
                            self.string += j
                            if data_word['conf'][num] <= 65:
                                if self.string[-1] in '!?,.:;':
                                    punkt = self.string[-1]
                                    self.string = self.string[:-1]
                                if number < 10 and number < max_len - 10:
                                    self.string = self.spell_check(self.string, ' '.join(new_mass_word[:self.count]))
                                elif number > 10 and number > max_len - 10:
                                    self.string = self.spell_check(self.string, ' '.join(
                                        new_mass_word[number - (self.count // 2 + (max_len - number)):]))
                                elif number > 10 and number < max_len - 10:
                                    self.string = self.spell_check(self.string, ' '.join(
                                        new_mass_word[number - self.count // 2:number + self.count // 2]))
                                else:
                                    self.string = self.spell_check(self.string, ' '.join(new_mass_word))
                                self.string += punkt
                            new_mass_word2.append(self.string)
                            self.string = ''
                    text = ' '.join(new_mass_word2)
                    text = text.replace('<', '')
                    text = text.replace('>', '')
                    self.FB2.text(text)
                else:
                    x, y, w, h = boxes.xywh.numpy()[n]
                    im_crop = im.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                    im_crop.save('out_img.jpg')
                    self.FB2.image('out_img.jpg')

    def end(self):
        path = self.FB2.end()
        return path

def create_fb2_full(path):
    model = MainClass()
    load = Loader(path)
    pages = load.len_pages()
    for i in range(pages):
        model.forward(load.forward_jpg())
        print(i, 'OK')
    path_fb2 = model.end()
    return path_fb2


"""yolo_weight = 'runs/detect/train3/weights/best.pt'
yolo_box_weight = 'runs/detect/train6/weights/best.pt'
bert_fold = 'bert_fol'
model = MainClass()
model.forward('my_data/77.jpg')
model.forward('my_data/78.jpg')
model.forward('my_data/79.jpg')
model.forward('my_data/80.jpg')
model.end()"""
create_fb2_full('/home/tor/PycharmProjects/Samsung/pdf_files/Opadchiy.pdf')