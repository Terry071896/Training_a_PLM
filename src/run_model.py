from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd

model_path = "/scratch/general/vast/u1427155/cs6966/assignment1/models/microsoft/deberta-v3-base-finetuned-imdb/checkpoint-12500"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
dataset = load_dataset('imdb')

sentence1_key, sentence2_key = ('text', None)
def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

counter = 0
failed = []
for i, data in enumerate(encoded_dataset['test']):
    # tokenized_text = tokenizer(data['text'],
    #                             truncation=True,
    #                             is_split_into_words=False,
    #                             return_tensors='pt')
    # print(tokenized_text)
    # print(torch.tensor([data['input_ids']]))
    outputs = model(torch.tensor([data["input_ids"]]))
    predicted_label = outputs.logits.argmax(-1)
    #print(predicted_label, data['label'])
    if predicted_label[0] != data['label']:
        counter +=1
        failed.append({'text' : data['text'], 'label' : data['label'], 'predicted' : predicted_label[0]})
        print(failed[-1])
    
    
df = pd.DataFrame(failed)
df.to_csv('/scratch/general/vast/u1427155/cs6966/assignment1/failed.csv')
