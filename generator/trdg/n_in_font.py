from fontTools.ttLib import TTFont
import os
import shutil


def char_in_font(unicode_char, font):
    for cmap in font['cmap'].tables:
        if cmap.isUnicode():
            if ord(unicode_char) in cmap.cmap:
                return True
    return False


with open('dicts/rub.txt', "r", encoding="utf8", errors="ignore") as d:
    lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
for i in os.listdir('fonts/latin'):
    fontpath = 'fonts/latin/' + i
    font = TTFont(fontpath)   # specify the path to the font in question
    mass = [char_in_font(char, font) for char in lang_dict]
    if all(mass):
        shutil.copyfile(fontpath, 'fonts/for_formul/' + i)


