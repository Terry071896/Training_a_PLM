from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import argparse
import numpy as np
import os

def fine_tune(output, 
            model_name="microsoft/deberta-v3-base", 
            hh_data='imdb', 
            hh_metric='accuracy',
            batch_size=2):

    dataset = load_dataset(hh_data)
    metric = load_metric(hh_metric)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    sentence1_key, sentence2_key = ('text', None)
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    args = TrainingArguments(
        os.path.join(output, f"{model_name}-finetuned-{hh_data}"),
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=hh_metric,
        push_to_hub=False,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        predictions = np.argmax(predictions, axis=1)
        
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print(trainer.evaluate())
    #trainer.predict(encoded_dataset["test"])
    #trainer.save_model(os.path.join(output, f"{model_name}-finetuned-{hh_data}_saved"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
    args = parser.parse_args()

    fine_tune(args.output_dir)