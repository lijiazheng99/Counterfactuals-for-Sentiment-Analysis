import logging
import torch

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Callable
from tqdm import tqdm

from cfsa.cfsa import Cfsa
from cfsa.constants import masker_sets, delimeters, lc_delimeters, punctuation
from copy import deepcopy
import regex as re

class Cfsarep(Cfsa):
    def __init__(self, texts, labels, neg_dict, pos_dict, neg_proun, finetuned_model, tokenizer, fill_model, mask_tokenizer, token_length, model_type):
        Cfsa.__init__(self, texts, labels, neg_proun, finetuned_model, tokenizer, token_length, model_type)
        self.neg_dict = neg_dict
        self.pos_dict = pos_dict
        self.fill_model = fill_model
        self.mask_tokenizer = mask_tokenizer
        self.masker = masker_sets[model_type]

    def get_sentiment_scores(self, plain_text, word_importance, predict_label):
        sentiment_embedding = []
        neg_proun_exist = False 
        neg_proun_existed = False
        for each in plain_text:
            if each in self.neg_proun:
                neg_proun_exist = True
                neg_proun_existed = True
            if neg_proun_exist:
                if each in self.pos_dict:
                    sentiment_embedding.append(0)
                elif each in self.neg_dict:
                    sentiment_embedding.append(1)
                else:
                    sentiment_embedding.append(-1)
                neg_proun_exist = False
            else:
                if each in self.pos_dict:
                    sentiment_embedding.append(1)
                elif each in self.neg_dict:
                    sentiment_embedding.append(0)
                else:
                    sentiment_embedding.append(-1)

        num_positive = sentiment_embedding.count(1)
        num_negative = sentiment_embedding.count(0)

        filtered_word_importance = []
        for importance, sentiment in zip (word_importance, sentiment_embedding):
            if sentiment == predict_label:
                filtered_word_importance.append(importance)
            else:
                filtered_word_importance.append(0)

        sorted_word_importance = sorted(filtered_word_importance, reverse = True)
        return sentiment_embedding, num_positive, num_negative, filtered_word_importance, sorted_word_importance, neg_proun_existed 

    def predict_masked(self, token_masked):
        input_ids = self.mask_tokenizer.encode(token_masked, add_special_tokens=False)
        masked_sen = self.mask_tokenizer.decode(input_ids)
        words = self.fill_model(masked_sen)
        replace_word = []
        for each in words:
            replace_word.append(self.mask_tokenizer.convert_ids_to_tokens(each['token']))
        return replace_word 

    def single_sentence_edit(self, sentence, label, LOGGER):  
        sentence = sentence.replace('...','... ')
        sentence = sentence.lower()
        predict_label = self.pred_sentence_sentiment(sentence)
        # If program failed in here, it probably because the sub-sentence length is 1. 
        # This could caused by a failure alignment of a sub-sentence.
        original_tokenized_text = self.tokenizer.tokenize(sentence)[:self.token_length-2]
        lower_tokenized_text = original_tokenized_text  

        last_time_carried = len(lower_tokenized_text)
        discard = False 

        while(last_time_carried != 0 and discard != True):
            plain_text = [text.replace(delimeters[self.current_model],'') for text in lower_tokenized_text]
            word_importance = self.get_word_importances(lower_tokenized_text)
            sentiment_embedding, num_positive, num_negative, filtered_word_importance, sorted_word_importance, neg_proun_existed = self.get_sentiment_scores(plain_text, word_importance, int(label))

            if label == '0' and num_negative == 0:
                if neg_proun_existed:
                    lower_tokenized_text = [each for each in lower_tokenized_text if each.replace(delimeters[self.current_model],'') not in self.neg_proun]
                    return False, self.tokenizer.decode(self.tokenizer.encode(lower_tokenized_text, add_special_tokens=False)), 1
                else:
                    return True, self.tokenizer.decode(self.tokenizer.encode(lower_tokenized_text, add_special_tokens=False)), 0
            elif label == '1' and num_positive == 0:
                if neg_proun_existed:
                    lower_tokenized_text = [each for each in lower_tokenized_text if each not in self.neg_proun]
                    return False, self.tokenizer.decode(self.tokenizer.encode(lower_tokenized_text, add_special_tokens=False)), 0
                else:
                    return True, self.tokenizer.decode(self.tokenizer.encode(lower_tokenized_text, add_special_tokens=False)), 0  

            for impor_score in sorted_word_importance:
                if impor_score > 0:
                    for idx, score in enumerate(filtered_word_importance):
                        if score == impor_score:
                            masked_tokens = deepcopy(lower_tokenized_text[:idx])
                            masked_tokens.append(self.masker)
                            masked_tokens.extend(lower_tokenized_text[idx+1:])
                            predict_words = self.predict_masked(masked_tokens)

                            if label == '1':
                                for each in predict_words:
                                    if each.lower().replace(lc_delimeters[self.current_model],'').replace(delimeters[self.current_model],'') in self.neg_dict:
                                        lower_tokenized_text[idx] = each
                                        num_positive -= 1
                                        break
                            elif label == '0':
                                for each in predict_words:
                                    if each.lower().replace(lc_delimeters[self.current_model],'').replace(delimeters[self.current_model],'') in self.pos_dict:
                                        lower_tokenized_text[idx] = each
                                        num_negative -= 1
                                        break
                            else:
                                LOGGER.warn('Predict label wrong in single_sentence_edit, please check the input label data type')

            if label == '1':
                if last_time_carried > num_positive and last_time_carried == len(lower_tokenized_text):
                    last_time_carried = num_positive
                elif last_time_carried != 0 and last_time_carried == num_positive:
                    discard = True
                    return discard, '', 0
                elif last_time_carried != num_positive:
                    last_time_carried = num_positive
            elif label == '0':
                if last_time_carried > num_negative and last_time_carried == len(lower_tokenized_text):
                    last_time_carried = num_negative
                elif last_time_carried != 0 and last_time_carried == num_negative:
                    discard = True
                    return discard, '', 0
                elif last_time_carried != num_negative:
                    last_time_carried = num_negative
            else:
                LOGGER.warn('Predict label wrong in single_sentence_edit, please check the input label data type')

        # new_predict_label never used, but can be used to validate and compare the difference of CF generation on original model.
        new_predict_label = self.pred_sentence_sentiment(lower_tokenized_text)
        input_ids = self.tokenizer.encode(lower_tokenized_text, add_special_tokens=False)
        decoded_text = self.tokenizer.decode(input_ids)

        return discard, decoded_text, new_predict_label

    def generate(self, output_path: str, LOGGER, store) -> pd.DataFrame:

        LOGGER.info('Start generating CF-REP for %s instances', len(self.texts))
        counterfactuals, original, part_original, new_labels = [], [], [], []   

        for idx, text in enumerate(tqdm(self.texts)):
            original_tokenized_text = self.tokenizer.tokenize(text)[:self.token_length-2]
            label = self.labels[idx]
            counterfactual_sentence, part_original_sentence, sub_sentence = '', '', []
            sentence_count = 0
            for inner_idx, each in enumerate(original_tokenized_text):
                if str(each) in punctuation or inner_idx == len(original_tokenized_text):
                    sub_sentence.append(each)
                    input_ids = self.tokenizer.encode(sub_sentence, add_special_tokens=False)
                    decoded_text = self.tokenizer.decode(input_ids)
                    inputed_sub_sentence = decoded_text
                    discard, decoded_text, new_predict_label = self.single_sentence_edit(decoded_text, label, LOGGER)
                    if discard and len(decoded_text) == 0:
                        pass
                    else:
                        part_original_sentence += inputed_sub_sentence 
                        part_original_sentence += ' '
                        counterfactual_sentence += decoded_text
                        counterfactual_sentence += ' '
                        sentence_count += 1
                    sub_sentence = []
                else:
                    sub_sentence.append(each)       

            if len(counterfactual_sentence) > 0 and sentence_count > 1:
                if label == '1' or label == '0':
                    new_labels.append(label)
                else:
                    LOGGER.warn('Generated a wrong new label')   

                counterfactuals.append(counterfactual_sentence)
                original.append(text)
                part_original.append(part_original_sentence)
                # Uncomment if you think watching progress bar is boring
                # print(counterfactual_sentence)
                # print(part_original_sentence)
            
            # Uncomment for a quick debug
            # if idx > 3:
            #     break

        LOGGER.info('Total %s CF-REP generated from %s original instances',len(counterfactuals) ,len(self.texts))
        df = pd.DataFrame()
        df['original'], df['part-original'], df['counterfact'], df['label'] = original, part_original, counterfactuals, new_labels
        if store:
            filename = f"{output_path}{str(self.current_model)}_{str(self.token_length)}_cfrep.csv"
            df.to_csv(Path(filename), index = False)
            LOGGER.info('CF-REP output successfully at %s ', filename)
        return df