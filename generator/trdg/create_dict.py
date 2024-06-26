import codecs
import re

def create_dict(path):
    reg = re.compile('[^a-zа-яA-ZА-Я0-9 ]')
    with open(path, 'r', encoding='UTF-8') as text:
        data = text.readlines()[101:180]
        for i in data:
            new_st = reg.sub('', i)
            while '  ' in new_st:
                new_st = new_st.replace('  ', '')
            if new_st != '':
                with open('/home/tor/PycharmProjects/pythonProject4/TextRecognitionDataGenerator/trdg/dicts/ab.txt',
                          'a', encoding='utf-8') as dictonary:
                    for word in new_st.split():
                        dictonary.write(word + '\n')


create_dict('Koltun_Solnechnye-elementy_RuLit_Me.txt')