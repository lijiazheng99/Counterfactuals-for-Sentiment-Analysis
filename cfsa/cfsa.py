import torch

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Callable
from tqdm import tqdm

from cfsa.constants import delimeters, lc_delimeters, punctuation
from copy import deepcopy
import regex as re

class Cfsa:
    def __init__(self, texts, labels, neg_proun, finetuned_model, tokenizer, token_length, model_type):
        self.texts = texts
        self.labels = labels
        self.neg_proun = neg_proun
        self.finetuned_model = finetuned_model
        self.tokenizer = tokenizer
        self.token_length = token_length
        self.current_model = model_type
    
    def pred_sentence_sentiment(self, sentence: str) -> str:

        if len(sentence) == 1:
            tokenized = self.tokenizer.tokenize(sentence)[:self.token_length-2]
        else:
            tokenized = sentence

        input_ids = torch.tensor([self.tokenizer.encode(tokenized, add_special_tokens=True)]).cuda()
        with torch.no_grad():
            predict_label = self.finetuned_model(input_ids)[0].argmax().item()
        return predict_label    

    def get_word_importances(self, tokens: list) -> list:

        unrecognizable = re.compile("#")
        importance = []

        with torch.no_grad():
            ori_logits = self.finetuned_model(torch.tensor([self.tokenizer.encode(tokens, add_special_tokens=True)]).cuda())[0]
            # ori_label = ori_logits.argmax().item()
        ori_neg, ori_pos = ori_logits[0][0].cpu().item(),ori_logits[0][1].cpu().item()

        for idx, token in enumerate(tokens):
            if len(str(token))>1 and len(unrecognizable.findall(token)) == 0:
                after_remove = deepcopy(tokens[:idx])
                after_remove.extend(tokens[idx+1:])
                input_ids = torch.tensor([self.tokenizer.encode(after_remove, add_special_tokens=True)]).cuda()
                with torch.no_grad():
                    removed_logits = self.finetuned_model(input_ids)[0]
                    # predict_label = logits.argmax().item()
                change_neg, change_pos = removed_logits[0][0].cpu().item(),removed_logits[0][1].cpu().item()
                change_value = change_neg - ori_neg
                importance.append(abs(change_value))
            else:
                importance.append(0)

        if len(importance)!=len(tokens):
            print('incorrect length')

        return importance 
    
    def remove_one_token_single(self, text_tokens: list):
        non = re.compile("#")
        candidiates_set = []
        removed_words = []
        for idx,token in enumerate(text_tokens):
            if len(str(token))>1 and len(non.findall(token))==0:
                removed_words.append(token.replace(delimeters[self.current_model],""))
                cleaned_list = [x for id_,x in enumerate(text_tokens) if id_ != idx ]
                candidiates_set.append(cleaned_list)
            else:
                continue
        return candidiates_set,removed_words
    
    def list_to_string(self, s:list) -> str:  
        str_ = ""  
        for idx, ele in enumerate(s):  
            str_ += ele  
            if idx != len(s)-1:
                str_ += " "
        return str_
    
    def intersection(self, lst1:list, lst2:list): 
        lst3 = [value for value in lst1 if value in lst2] 
        return lst3