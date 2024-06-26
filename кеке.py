from gensim.models import FastText, keyedvectors
from gensim.test.utils import datapath
from gensim.test.utils import common_texts  # some example sentences
from transformers import AutoModel, AutoTokenizer
from spellchecker import SpellChecker

spell = SpellChecker(language='ru', distance=2)
spell.word_frequency.load_dictionary('/home/tor/PycharmProjects/Samsung/Spell/data/ru_full.json')

print(spell.correction('коллекюра'))


s = keyedvectors.FastTextKeyedVectors.load('/home/tor/PycharmProjects/Samsung/fast_text_fold/model.model')
print(s.similarity('Нанболее', 'более'))
