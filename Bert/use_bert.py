from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('../bert_fine_tune/checkpoint-1500')
config = 'bert_fine_tune/checkpoint-1500/config.json'
model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path='bert_fol', config='bert_fol/config.json')

print('OK')

from transformers import pipeline
config = '../bert_fine_tune/checkpoint-1500/config.json'
unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
print('рост толщины базы дополнительно увеличивает ток коллек- юра К2, а также уменьшает ток коллектора К1. Это приводит к дополнительному росту магниточувствительности ДМТ.')
print(unmasker("рост толщины базы дополнительно увеличивает ток [MASK] К2, а также уменьшает ток коллектора К1. Это приводит к дополнительному росту магниточувствительности ДМТ.")[0]['sequence'])
for i in unmasker("рост толщины базы дополнительно увеличивает ток коллектора К2, а также уменьшает ток коллектора К1. Это приводит к [MASK] росту магниточувствительности ДМТ."):
    print(i['token_str'])