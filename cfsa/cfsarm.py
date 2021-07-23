import logging
import torch
import math

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Callable
from tqdm import tqdm

from cfsa.cfsa import Cfsa
from cfsa.constants import masker_sets, delimeters, lc_delimeters
from copy import deepcopy
import regex as re

class Cfsarm(Cfsa):
    def __init__(self, texts, labels, neg_proun, finetuned_model, tokenizer, token_length, model_type):
        Cfsa.__init__(self, texts, labels, neg_proun, finetuned_model, tokenizer, token_length, model_type)
    
    def single_sentence_edit(self, inputs):
 
        input_text = self.tokenizer.tokenize(inputs)[:self.token_length-2]
        current_tokens = deepcopy(input_text)
        input_ids = torch.tensor([self.tokenizer.encode(input_text, add_special_tokens=True)]).cuda()

        with torch.no_grad():
            ori_logits = self.finetuned_model(input_ids)[0]  # Models outputs are now tuples
            ori_pred_label = ori_logits.argmax().item()

        reduced_instances,removed_words = self.remove_one_token_single(input_text) #return candidate list 

        logits_changed,label_after = [], []

        for idx,text in enumerate(reduced_instances):
            input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)]).cuda()
            with torch.no_grad():
                logits = self.finetuned_model(input_ids)[0]  # Models outputs are now tuples
                pred_label = logits.argmax().item()

            change_neg, change_pos = logits[0][0].cpu().item(),logits[0][1].cpu().item()
            ori_neg, ori_pos = ori_logits[0][0].cpu().item(),ori_logits[0][1].cpu().item()

            changes_value = change_neg-ori_neg
            logits_changed.append(changes_value)
            label_after.append(pred_label)

        label_changed_id = [idx for idx, value in enumerate(label_after) if value!=ori_pred_label]

        if len(label_changed_id) != 0:
            # print("**** Thanks god, label change here at: ",label_changed_id)
            logit = []
            for ids in label_changed_id:
                logit.append(logits_changed[ids])

            change_max = max(min(logit), max(logit), key=abs)
            # print("**** The max change is: ", change_max)

            id_max = [i for i, j in enumerate(logit) if j == change_max]
            id_max = [label_changed_id[id_m] for id_m in id_max]
            id_max_plus = [id_max[0] + i for i in range(3)]
        else:
            change_max = max(min(logits_changed), max(logits_changed), key=abs)
            # print("**** The max change is: ",  change_max)
            id_max = [i for i, j in enumerate(logits_changed) if j == change_max]
            id_max_plus = [id_max[0] + i for i in range(3)]

        text_found = [reduced_instances[id_] for id_ in id_max]
        label_changed_word = [removed_words[id_] for id_ in id_max if label_after[id_]!=ori_pred_label]
        try:
            removed_words_plus = [removed_words[i].lower() for i in id_max_plus]
        except:
            try:
                removed_words_plus = [removed_words[i].lower() for i in id_max_plus[:2]]
            except:
                removed_words_plus=[]
        removed_words = [removed_words[id_].lower() for id_ in id_max]
        removed_words = list(set(removed_words))

        if ((removed_words[0] in self.neg_proun) and len(removed_words_plus)>0):
            # print('neg_proun is here', removed_words_plus)
            removed_words = removed_words_plus

        # print ("**** The word removed here is: ",removed_words)
        return label_changed_id, removed_words, change_max, id_max, text_found, inputs
    
    def generate(self, output_path: str, LOGGER, store) -> pd.DataFrame:
        LOGGER.info('Start generating CF-RM for %s instances', len(self.texts))
        dict_words_pos, dict_words_neg = {}, {}
        removed_words_corpus, text_found_corpus, change_max_corpus = [], [], []
        position_max_corpus, label_changed_word_corpus, ori_inputs = [], [], []

        for idx, test in enumerate(tqdm(self.texts)):

            label_changed_id, removed_words, change_max, id_max, text_found, ori_input = \
            self.single_sentence_edit(test)
            words_str = self.list_to_string(removed_words)
            if change_max >= 0:
                if words_str not in list(dict_words_pos.keys()):  
                    d = {words_str: abs(change_max)}
                    dict_words_pos.update(d)
                else:
                    dict_words_pos[words_str] += abs(change_max)
            else:
                if words_str not in list(dict_words_neg.keys()):
                    d = {words_str: abs(change_max)}
                    dict_words_neg.update(d)
                else:
                    dict_words_neg[words_str] += abs(change_max)        

            removed_words_corpus.append(removed_words)
            text_found_corpus.append(text_found)
            position_max_corpus.append(id_max)
            change_max_corpus.append(change_max)
            ori_inputs.append(ori_input)
            if len(label_changed_id) != 0:
                label_changed_word_corpus.append(idx)

        preds = [self.pred_sentence_sentiment(each) for each in self.texts]
        wrong_predictions = [i for i in range(len(preds)) if int(preds[i])!=int(self.labels[i])]
        frail_instances = self.intersection(wrong_predictions, label_changed_word_corpus)

        new_instances, removed, new_labels, changed_instances = [], [], [], []
        for i, val in enumerate(self.texts):
            if i in label_changed_word_corpus and i not in frail_instances:
                val_tokens = self.tokenizer.tokenize(val)[:self.token_length-2]
                assert len(val_tokens) == len(text_found_corpus[i][0])+1

                input_ids = self.tokenizer.encode(text_found_corpus[i][0], add_special_tokens=False)
                new_instances.append(self.tokenizer.decode(input_ids))
                changed_instances.append(val)
                if self.labels[i] == '0':
                    new_labels.append('1')
                else:
                    new_labels.append('0')
                for idx in range(len(val_tokens)):
                    if idx!=len(val_tokens)-1:
                        if val_tokens[idx] != text_found_corpus[i][0][idx]:
                            removed.append(val_tokens[idx])
                            break
                    else:
                        removed.append(val_tokens[idx])

        LOGGER.info('Total %s CF-RM generated from %s original instances',len(new_instances) ,len(self.texts))
        df = pd.DataFrame()
        df['original'], df['counterfact'], df['label'] = changed_instances, new_instances, new_labels
        if store:
            filename = f"{output_path}{str(self.current_model)}_{str(self.token_length)}_cfrm.csv"
            df.to_csv(Path(filename), index = False)
            LOGGER.info('CF-RM output successfully at %s ', filename)
        return df