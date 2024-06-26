from spellchecker import SpellChecker
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
model = AutoModelForMaskedLM.from_pretrained('DeepPavlov/rubert-base-cased')
print('OK')

from transformers import pipeline
unmasker = pipeline('fill-mask', model='DeepPavlov/rubert-base-cased')
print(unmasker("в пределах которого работает нелинейный элемент, известен [MASK] может быть аппроксимирован прямой без излома. В этом случае нелинейный резистивный элемент заменяется источником"))
