import sys

import pandas as pd
import numpy as np

from cfsa.constants import models_dict, tokenizers_dict, mlm_settings
from pathlib import Path

from transformers import AutoTokenizer, pipeline

def dataset_loader(dataset_file_path: str):
    """
    Attempts to load dataset via pandas csv reader from a given file path. File path should use forward slashes, regardless of OS
    Parameters. Default path is loading an dataset from github repositry.
    ----------
    dataset_file_path: str
        Path to the dataset
    Returns
    -------
    texts: list
    labels: list
    """

    # You may need to customize read_csv arguments here if you want to use a different dataset
    try:
        dataset = pd.read_csv(dataset_file_path, delimiter='\t')
    except(FileNotFoundError):
        LOGGER.warn('File not found %s', dataset_file_path)
        sys.exit(-1)
    # LOGGER.info('Successfully loaded dataset')

    # You may want to modify this part to fit your dataset
    texts = dataset['Text'].to_list()
    labels = dataset['Sentiment'].to_list()
    labels = ['1' if _=='Positive' else _ for _ in labels]
    labels = ['0' if _=='Negative' else _ for _ in labels]

    # if len(texts) == len(labels):
    #     LOGGER.info('Successfully loaded dataset')
    # else:
    #     LOGGER.error('Different length of text and label data')
    #     raise Exception

    return texts, labels

def dict_loader(dictionary_path: str):
    """
    Attempts to load positive/negtive dictionary from a given file path.
    ----------
    dictionary_path: str
        Path to the dataset
    Returns
    -------
    texts: list
    labels: list
    """

    # You may want to customize this part to load your selected dictionary
    try:
        with open(Path(dictionary_path + 'negative.txt'),'r', encoding = "ISO-8859-1") as f:
            neg_dict = f.readlines()
    except(FileNotFoundError):
        sys.exit(-1)

    neg_dict = [each.replace('\n','') for each in neg_dict]

    try:
        with open(Path(dictionary_path + 'positive.txt'),'r', encoding = "ISO-8859-1") as f:
            pos_dict = f.readlines()
    except(FileNotFoundError):
        sys.exit(-1)

    pos_dict = [each.replace('\n','') for each in pos_dict]

    return neg_dict, pos_dict

def neg_proun_loader(dictionary_path: str):
    return np.load(Path(dictionary_path + 'neg_proun.npy'))

def model_loader(model_path: str, tokenizer_path: str, model_type: str):
    """
    Attempts to load fine-tuned language model and masked language model via defined model path and model type.
    ----------
    model_path: str
        Path to the best checkpoint
    tokenizer_path: str
        Path to the tokenizer
    model_type: str
        model types Bert, Roberta, Xlnet
    Returns
    -------
    finetuned_model: language model
    tokenizer: tokenzier
    fill_model: language model
    mask_tokenizer: tokenizer
    """
    finetuned_model = models_dict[model_type](model_path)
    tokenizer = tokenizers_dict[model_type](tokenizer_path)
    finetuned_model.cuda()

    # Docs on pipeline fucntion: https://huggingface.co/transformers/main_classes/pipelines.html?highlight=pipeline%20load#pipelines
    mask_tokenizer = AutoTokenizer.from_pretrained(mlm_settings[model_type])
    fill_model = pipeline('fill-mask', model=mlm_settings[model_type])

    return finetuned_model, tokenizer, fill_model, mask_tokenizer 