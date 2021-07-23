# Moverscore: https://github.com/AIPHES/emnlp19-moverscore
# Use the original version with BERTMNLI to reproduce the results.
# from moverscore import get_idf_dict, word_mover_score
# Recommend to use this version (DistilBERT) for evaluation, if the speed is your concern.
from moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict

import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List

class MissingColumnException(Exception):
    __module__ = Exception.__module__

class Mvscore:
    def __init__(self, df:pd.DataFrame, threshold:int):
        self.df = df
        self.threshold = threshold
    
    def check_df(self, LOGGER) -> bool:
        for each in ['original', 'counterfact', 'label']:
            if each in self.df.columns:
                pass
            else:
                LOGGER.error('Counterfactual dataframe missing %s in Moverscore selection process', each)
                raise MissingColumnException
        return True
    
    def by_word_mover_score(self, original:str, counterfact:str):
        idf_dict_hyp = get_idf_dict(original)
        idf_dict_ref = get_idf_dict(counterfact)

        scores = word_mover_score(counterfact, original, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True, batch_size=4)
        return scores[0]

    def sentence_score(self, hypothesis: str, references: List[str], trace=0):
        # https://github.com/AIPHES/emnlp19-moverscore/blob/master/examples/example.py

        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)

        hypothesis = [hypothesis] * len(references)

        sentence_score = 0 

        scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)

        sentence_score = np.mean(scores)

        if trace > 0:
            print(hypothesis, references, sentence_score)

        return sentence_score

    def selection(self, args, LOGGER) -> None:
        if self.check_df(LOGGER):
            originals, counterfacts, labels = [], [], []
            for idx, row in tqdm(self.df.iterrows()):
                original = row['original']
                counterfact = row['counterfact']
                score = self.sentence_score(counterfact, [original])
                # score = self.by_word_mover_score([counterfact], [original])
                
                if score > self.threshold:
                    originals.append(original)
                    counterfacts.append(counterfact)
                    labels.append(row['label'])
                else:
                    pass
            
            df = pd.DataFrame()
            df['original'], df['counterfact'], df['label'] = originals, counterfacts, labels
            filename = f"{args.output_path}{str(args.model_type)}_{str(args.token_length)}_selected.csv"
            df.to_csv(Path(filename), index = False)
            LOGGER.info('Moverscore selected counterfacts output successfully at %s ', filename)

def combine_dfs(df_rep: pd.DataFrame, df_rm: pd.DataFrame, args) -> pd.DataFrame:
    """
    Combine CF-REP & CF-RM as one dataframe
    ----------
    df_rep: DataFrame
        CF-REP
    df_rm: DataFrame
        CF-RM
    Returns
    -------
    df_cbm: DataFrame
        combination of CF-REP and CF-RM 
    """
    originals, counterfactuals, labels = [], [], []
    originals.extend(df_rep['part-original'])
    originals.extend(df_rm['original'])
    counterfactuals.extend(df_rep['counterfact'])
    counterfactuals.extend(df_rm['counterfact'])
    labels.extend(df_rep['label'])
    labels.extend(df_rm['label'])
    df = pd.DataFrame()
    df['original'] = originals
    df['counterfact'] = counterfactuals
    df['label'] = labels
    if args.store_cfs:
        filename = f"{args.output_path}{str(args.model_type)}_{str(args.token_length)}_combined.csv"
        df.to_csv(Path(filename), index = False)
    return df