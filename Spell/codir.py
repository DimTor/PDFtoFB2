import os

from chardet.universaldetector import UniversalDetector
import codecs
import re


string = ''
books = ['book.txt', 'book2.txt', 'book3.txt', 'book4.txt', 'book5.txt', 'book6.txt', 'book7.txt']
for book in os.listdir(os.getcwd()):
    if book.split('.')[-1] != 'txt':
        continue
    detector = UniversalDetector()
    with open(book, 'rb') as fh:
        for line in fh:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    enc = detector.result['encoding']

    fh = codecs.open(book, 'r', enc)
    s = fh.read()
    cleanString = re.sub('[^а-яА-Я!?*(){}\-:;<>.,''"" ]', ' ', s)
    while '  ' in cleanString:
        cleanString = cleanString.replace('  ', ' ')
    while '..' in cleanString:
        cleanString = cleanString.replace('..', '.')
    string += cleanString + ' '


with open('my_big_txt.txt', 'w', encoding='utf-8') as my:
    my.write(string)