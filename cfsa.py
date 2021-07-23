import sys
import argparse
import logging
import os

from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

import torch

from transformers import *


from cfsa.constants import TRAIN_SET_URL, DICT_PATH, models_dict
from cfsa.loader import dataset_loader, dict_loader, neg_proun_loader, model_loader
from cfsa.cfsarep import Cfsarep
from cfsa.cfsarm import Cfsarm

VERSION = '1.0'

os.makedirs(os.path.dirname('outputs/'), exist_ok=True)

now = datetime.now()
file_name = Path(f'outputs/{now.strftime("%d_%m_%Y_%H.%M.%S")}.log')
logging.basicConfig(filename=file_name, filemode='w', \
                    format='%(asctime)s, %(levelname)s %(message)s',\
                    datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
LOGGER = logging.getLogger('CfSA')

# Too lazy to create customize exception
class ModelTypeError(Exception):
    __module__ = Exception.__module__

class DatasetError(Exception):
    __module__ = Exception.__module__

def init_argparser() -> argparse.ArgumentParser:
    """
        Creates an argument parser with all of the possible
        command line arguments that can be passed to CFSA
    """
    parser = argparse.ArgumentParser(description="Conterfactuals-for-sentiment-analysis")

    required = parser.add_argument_group('Required Arguments')
    required.add_argument('--train_set', required=False, type=str, default=TRAIN_SET_URL, dest='train_path')
    required.add_argument('--dict_path', required=False, type=str, default=DICT_PATH)

    required.add_argument('--model', required=False, type=str, default='bert-base', dest='model_type')
    required.add_argument('--token_length', required=False, type=int, default=128)
    required.add_argument('--best_model', required=True, dest='best_model_path')
    required.add_argument('--tokenizer', required=True, dest='tokenizer_path')

    required.add_argument('--output_path', required=False, default='outputs/', dest='output_path')

    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('--cuda', type=str, default='0', dest='cuda_gpu')
    optional.add_argument('--store_cfs', action='store_true', default=False)
    optional.add_argument('--debug', action='store_true', default=False)
    optional.add_argument('-v', '--version', action='version', version=f'{parser.prog} version {VERSION}')

    return parser

def main() -> None:
    # Please enjoy Jiazheng's terrible code, 
    # welcome any complaints to improve my coding skills.

    parser = init_argparser()
    args = parser.parse_args()

    LOGGER.setLevel(logging.DEBUG if args.debug else logging.INFO)

    if args.model_type in models_dict.keys():
        LOGGER.info('Model type valid %s', args.model_type)
    else:
        LOGGER.error('Model type invalid %s', args.model_type)
        raise ModelTypeError("Model type error. Program only takes four types of models:" + str(models_dict.keys()))

    LOGGER.info('Current settings: Dataset path %s, Dictioinary path %s, Output path %s, Store %s', args.train_path, args.dict_path, args.output_path, args.store_cfs)
    LOGGER.info('Model type %s, Token length %s, Best model %s, Tokenizer %s, Cuda device %s', args.model_type, args.token_length, args.best_model_path, args.tokenizer_path, args.cuda_gpu)

    LOGGER.info('Set CUDA device as %s', args.cuda_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_gpu)
    LOGGER.info('CUDA device set successfully')
    
    # Ensure you have connected to internet
    LOGGER.info('Loading dataset from %s', args.train_path)
    texts, labels = dataset_loader(args.train_path)
    LOGGER.info('Successfully loaded dataset from %s, texts length: %s, labels length: %s', args.train_path, len(texts), len(labels))

    # Load dictionary
    LOGGER.info('Loading dictionaries from %s', args.dict_path)
    neg_dict, pos_dict = dict_loader(args.dict_path)
    neg_proun = neg_proun_loader(args.dict_path)
    LOGGER.info('Successfully loaded dictionaries from %s, negtive dictionary length: %s, postive dictionary length: %s', args.dict_path, len(neg_dict), len(pos_dict))

    # Load fune-tuned model and masked language model
    LOGGER.info('Loading model from %s, tokenizer from %s, model type as %s', args.best_model_path, args.tokenizer_path, args.model_type)
    finetuned_model, tokenizer, fill_model, mask_tokenizer = model_loader(args.best_model_path, args.tokenizer_path, args.model_type)
    LOGGER.info('Successfully loaded models')

    # Generate REP and REP+RM instances
    LOGGER.info('Generating REP counterfactuals')
    cfsarep = Cfsarep(texts, labels, neg_dict, pos_dict, neg_proun, finetuned_model, tokenizer, fill_model, mask_tokenizer, args.token_length, args.model_type)
    LOGGER.info('CF-REP class initialize successfully')
    df_rep = cfsarep.generate(args.output_path, LOGGER, args.store_cfs)

    # Generate RM instances
    LOGGER.info('Generating RM counterfactuals')
    cfsarm = Cfsarm(texts, labels, neg_proun, finetuned_model, tokenizer, args.token_length, args.model_type)
    LOGGER.info('CF-RM class initialize successfully')
    df_rm = cfsarm.generate(args.output_path, LOGGER, args.store_cfs)

    from mvscore.mvscore import combine_dfs, Mvscore

    # Select counterfactuals with Moverscore
    LOGGER.info('Select counterfactuals with moverscore')
    mvscore = Mvscore(combine_dfs(df_rep, df_rm, args), 0.55)
    LOGGER.info('Moverscore class initialize successfully')
    mvscore.selection(args, LOGGER)


if __name__ == '__main__':
    main()