import pandas as pd
import numpy as np
import random
import torch
import transformers
import torch.nn as nn
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer
import evaluate
from torch.utils.data import Dataset
import evaluate
from transformers import DataCollatorForLanguageModeling
from transformers import default_data_collator
import collections
wwm_probability = 0.1

def whole_word_masking_data_collator(features):
    for feature in features:
        mask = np.random.binomial(1, wwm_probability, 256,)
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            if input_ids[word_id] != 0:
                new_labels[word_id] = labels[word_id]
                input_ids[word_id] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

class CustomDataset(Dataset):
    def __init__(self, tokenizer, n, e):
        dataset = pd.read_csv('/home/tor/PycharmProjects/Samsung/Bert/bert_fine_tune/out.csv')
        dataset = dataset['text'][n:e].copy().reset_index(drop=True)
        print(dataset[0])
        self.tokenizer = tokenizer
        self.token_dataset = dataset.map(self.tokenize_function)
        self.token_dataset = self.token_dataset.map(self.group_texts)

    def tokenize_function(self, examples):
        result = self.tokenizer(examples, padding="max_length", max_length=128, truncation=True)
        return result

    def group_texts(self, examples):
        # Concatenate all texts
        # Create a new labels column
        examples['labels'] = examples['input_ids'].copy()
        return examples


    def __len__(self):
        return len(self.token_dataset)

    def __getitem__(self, idx):
        return self.token_dataset[idx]


tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
model = AutoModelForMaskedLM.from_pretrained('DeepPavlov/rubert-base-cased')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

train_dataset = CustomDataset(tokenizer, 0, 7000)
test_dataset = CustomDataset(tokenizer, 7200, 7500)

samples = [train_dataset[i] for i in range(2)]
#batch = whole_word_masking_data_collator(samples)

from transformers import TrainingArguments

batch_size = 20
# Show the training loss with every epoch
logging_steps = len(train_dataset) // batch_size

training_args = TrainingArguments(
    output_dir="bert_fine_tune",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    logging_steps=logging_steps,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model('bert_fol')
model.save_pretrained("bert_folder")
tokenizer.save_pretrained("bert_folder")