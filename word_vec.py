from gensim.models import Word2Vec
import gensim
from sup_nlp_tools import tokenize_corpus
import gensim.downloader as api
import numpy as np
import pymorphy2
from gensim.models import FastText

pretrained = Word2Vec.load('/home/tor/PycharmProjects/Samsung/fast_text_fold/model.model')
morph = pymorphy2.MorphAnalyzer()
word = 'нанболее'
parsed_word = morph.parse(word)[0]
print(f"Нормальная форма: {parsed_word.normal_form}")
print(f"Часть речи: {parsed_word.tag.POS}")
word = parsed_word.normal_form + '_' + parsed_word.tag.POS

print(pretrained.most_similar(word))

