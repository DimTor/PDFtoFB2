from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import pipeline
import os

os.environ["TOKENIZERS_PARALLELISM"] = "екгу"


class BERTClass:
    def __init__(self, bert_fold):
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=bert_fold,
                                                     config=f'{bert_fold}/config.json')
        self.tokenizer = AutoTokenizer.from_pretrained(bert_fold)
        self.unmasker = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer)

    def predict(self, text: str, words):
        new_text = text
        for word in words:
            new_text = new_text.replace(word, '[MASK]')
        try:
            res = self.unmasker(new_text)
            if type(res[0]) is list:
                for n, r in enumerate(res):
                    text = text.replace(words[n], r[0]['token_str'])
            else:
                return res[0]['token_str']
        except Exception:
            return words[0]


#s = BERTClass('bert_fol')
#print(s.predict("Нанболее просто эта задача решается в частбом случае, когда нелинейность характеристики мала", ['Нанболее', 'частбом']))