import os
import base64


class FB2:
    def __init__(self, title='пробная книга', name='Дмитрий', surname='Торотенков', lang='ru', annotation='', cover='117_90_Logo-MEPHI.jpg'):
        with open('binary_txt.txt', 'wb'):
            pass
        self.imgs_id = 1
        self.title = title.replace(' ', '_')
        with open('Draft.txt', 'r') as draft:
            content = draft.read()
        content = content.replace('{name_author}', name)
        content = content.replace('{surname_author}', surname)
        content = content.replace('{title}', title)
        content = content.replace('{lang}', lang)
        content = content.replace('{annotation}', annotation)
        with open('My_testbook.txt', 'w') as test_book:
            test_book.write(content + '<body><section>')
        image = open(cover, 'rb').read()
        with open('binary_txt.txt', 'a') as test_book:
            content = f'<binary id="cover.jpg" content-type="image/jpeg">{base64.encodebytes(image).decode()}\n</binary>'
            test_book.write(content)

    def text(self, text):
        with open('My_testbook.txt', 'a') as test_book:
            content = '<p>' + text + '</p>'
            test_book.write(content)

    def image(self, image_path):
        with open('My_testbook.txt', 'a') as test_book:
            content = f'<image xlink:href="#{self.imgs_id}.jpg" />'
            test_book.write(content)
        image = open(image_path, 'rb').read()
        with open('binary_txt.txt', 'a') as test_book:
            content = f'<binary id="{self.imgs_id}.jpg" content-type="image/jpeg">{base64.encodebytes(image).decode()}</binary>'
            test_book.write(content)
        self.imgs_id += 1

    def end(self):
        with open('binary_txt.txt', 'r') as binary:
            bin_imgs = binary.read()
        with open('My_testbook.txt', 'a') as test_book:
            content = '</section></body>' + bin_imgs + '</FictionBook>'
            test_book.write(content)
        os.rename('My_testbook.txt', f'{self.title}.fb2')
        return f'{self.title}.fb2'



"""fb2 = FB2('МИФИ23', 'Дмиhтрий', 'Торотенjков', 'ru')
fb2.text('икзлиымххлилкхлхи лизлиылизлиизылк')
fb2.image('datasets/coco8/images/val/000000000036.jpg')
fb2.text('едкдихфджхмдхмффд пдкдхдкхмдм')
fb2.image('datasets/coco8/images/val/000000000036.jpg')
fb2.text('едкдихфджхмдхмффд пдкдхдкхмдм')
fb2.end()"""
