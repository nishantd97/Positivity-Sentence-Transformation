import argparse
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
import os
from transformers import Trainer, pipeline, set_seed, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
nltk.download('punkt')
import csv
from datasets import load_dataset, load_metric
from utils.utils_metric import preprocess_function, compute_metric_with_extra

def run_t5_unconstrained(): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    metric3 = load_metric('bertscore')
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #print(tokenizer("Hello, this one sentence!")) #a test
    prefix = "summarize: "

    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_text", "type_label":"reframed_text"})
   
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_text", "type_label": "reframed_text"})
      
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    compute_metric = compute_metric_with_extra(tokenizer, metric, metric2, metric3)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metric
    )
    trainer.train()
    trainer.evaluate()
    # save model
    trainer.save_model(os.path.join(model_path,'t5_unconstrained/'))



def run_t5_controlled(): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    metric3 = load_metric('bertscore')
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    prefix = "summarize: "

    
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_with_label", "type_label":"reframed_text"})
   
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_with_label", "type_label":"reframed_text"})
      

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    compute_metric = compute_metric_with_extra(tokenizer, metric, metric2, metric3)

    
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metric
    )
    trainer.train()
    trainer.evaluate()
    # save model
    trainer.save_model(os.path.join(model_path,'t5_controlled/'))


def run_t5_predict(): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    metric3 = load_metric('bertscore')
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #print(tokenizer("Hello, this one sentence!")) #a test
    prefix = "summarize: "

    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_text", "type_label":"strategy_reframe"})
   
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_text", "type_label": "strategy_reframe"})
    

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    compute_metric = compute_metric_with_extra(tokenizer, metric, metric2, metric3)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metric
    )

    trainer.train()
    trainer.evaluate()
    # save model
    trainer.save_model(os.path.join(model_path,'t5_predict/'))




def main():
    #run models
    if args.setting =='unconstrained':
        run_t5_unconstrained()
    elif  args.setting =='controlled':
        run_t5_controlled()
    elif args.setting =='predict':
        run_t5_predict()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', default='unconstrained', choices=['unconstrained', 'controlled', 'predict'])
    parser.add_argument('--train', default='data/wholetrain.csv') #default is for bart/t5; data format will be different for GPT
    parser.add_argument('--dev', default='data/wholedev.csv')
    parser.add_argument('--test', default='data/wholetest.csv')
    parser.add_argument('--model_dir', type=str, default='models')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    train_path = args.train
    train_dataset = load_dataset('csv', data_files=train_path)
    dev_path = args.dev
    dev_dataset = load_dataset('csv', data_files=train_path)
    test_path = args.test
    test_dataset = load_dataset('csv', data_files=test_path)

    model_path = args.model_dir
    main()