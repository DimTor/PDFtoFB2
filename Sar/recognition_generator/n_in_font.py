from fontTools.ttLib import TTFont
import os


def char_in_font(unicode_char, font):
    for cmap in font['cmap'].tables:
        if cmap.isUnicode():
            if ord(unicode_char) in cmap.cmap:
                return True
    return False


for i in os.listdir('fonts/latin'):
    fontpath = 'fonts/latin/' + i
    font = TTFont(fontpath)   # specify the path to the font in question
    mas = (char_in_font('γ', font)),
    #if not char_in_font('№', font):
    #    os.remove(fontpath)


