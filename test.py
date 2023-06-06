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

def run_t5_unconstrained():
    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_path,'t5_unconstrained/'))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path,'t5_unconstrined/'))
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_unconstrained.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)



def run_t5_controlled(): 
    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_path,'t5_controlled/'))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path,'t5_controlled/'))
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_with_label'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_controlled.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)


def run_t5_predict(): 
    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_path,'t5_predict/'))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path,'t5_predict/'))
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_predict.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)




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
    parser.add_argument('--test', default='data/wholetest.csv')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--model_dir', type=str, default='models')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    test_path = args.test
    test_dataset = load_dataset('csv', data_files=test_path)

    path = args.output_dir
    model_path = args.model_dir
    main()