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

def preprocess_function(examples, tokenizer, prefix, type, type_label):
        inputs = [prefix + doc for doc in examples[type]]
        model_inputs = tokenizer(inputs) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[type_label]) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def compute_metric_with_extra(tokenizer, metric, metric2, metric3):
    tokenizer = tokenizer
    metric = metric
    metric2 = metric2
    metric3 = metric3
    def compute_metric(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

        # print(result2)
        result['sacrebleu'] = round(result2["score"], 1)
        

        result3 = metric3.compute(predictions=decoded_preds, references=decoded_labels_expanded,  lang="en")

        result['bertscore'] = sum(result3["f1"])/len(result3["f1"])

        return {k: round(v, 6) for k, v in result.items()}
    return compute_metric
