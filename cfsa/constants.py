from transformers import *

TRAIN_SET_URL = 'https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master/sentiment/orig/train.tsv'
DICT_PATH = 'sa_dictionary/'

punctuation = '!.?'

models_dict = {'roberta': RobertaForSequenceClassification.from_pretrained,
               'xlnet': XLNetForSequenceClassification.from_pretrained,
               'bert-large': BertForSequenceClassification.from_pretrained,
               'bert-base': BertForSequenceClassification.from_pretrained}

tokenizers_dict = {'roberta' : RobertaTokenizer.from_pretrained,
                   'xlnet': XLNetTokenizer.from_pretrained,
                   'bert-large': BertTokenizer.from_pretrained,
                   'bert-base': BertTokenizer.from_pretrained}

# Check https://huggingface.co/models?filter=masked-lm to explore different masked language model combinations.
mlm_settings = {'roberta': 'roberta-large',
                'xlnet': 'xlnet-large-cased',
                'bert-large': 'bert-large-uncased',
                'bert-base': 'bert-base-uncased'}

masker_sets = {'roberta':'<mask>', 'xlnet': '<mask>', 'bert-large': '[MASK]', 'bert-base': '[MASK]'}

delimeters = {'roberta': 'Ġ', 'xlnet': '▁', 'bert-large': '', 'bert-base': ''}
lc_delimeters = {'roberta': delimeters['roberta'].lower(), 'xlnet': delimeters['xlnet'].lower(), 'bert-large': '', 'bert-base': ''}