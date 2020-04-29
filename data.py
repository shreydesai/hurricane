import collections

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, RobertaTokenizer
import spacy

from main import args


if args.tokenizer == 'word':
    spacy_tokenizer = spacy.load('en')
elif args.tokenizer == 'bert':
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif args.tokenizer == 'roberta':
    bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

emotions = [
    'aggressiveness',
    'optimism',
    'love',
    'submission',
    'awe',
    'disapproval',
    'remorse',
    'contempt',
]
emotions_enc = {e: i for i, e in enumerate(emotions)}


def _process_tokens(tokens):
    for i in range(len(tokens)):
        if tokens[i].startswith('http'):
            tokens[i] = '<url>'
        elif tokens[i].startswith('@'):
            tokens[i] = '<user>'
    return tokens


def tokenize_char(sent):
    return list(sent)


def tokenize_word(sent):
    return sent.split()


def tokenize_subword(sent):
    return bert_tokenizer.tokenize(sent)


def build_vocab(args, tokenizer):
    vocab = collections.Counter()
    df = pd.read_csv(args.train_path)
    for i, row in df.iterrows():
        tokens = _process_tokens(tokenizer(row['text']))
        vocab.update(tokens)
    words = ['<pad>', '<unk>', '<bos>', '<eos>'] + list(sorted(vocab))
    return (
        words,
        {w: i for i, w in enumerate(words)},
    )


class HurricaneDataset(Dataset):
    def __init__(self, args, ds_path):
        self.df = pd.read_csv(self._get_path(args, ds_path))
        self.tokenizer = self._get_tokenizer(args)
        self.msl = args.max_seq_len
        self.use_bert_encoder = (
            args.model == 'bert' or args.model == 'roberta'
        )
        if not self.use_bert_encoder:
            self.vocab, self.enc = build_vocab(args, self.tokenizer)
        self.device = args.device
        self.pad_idx = args.pad_idx
        self.unk_idx = args.unk_idx

        self._label = [x for x in list(self.df) if x != 'text'].pop()
        self._cache = {}

    def _get_path(self, args, ds_path):
        if ds_path == 'train':
            return args.train_path
        elif ds_path == 'valid':
            return args.valid_path
        elif ds_path == 'test':
            return args.test_path
        raise NotImplementedError

    def _get_tokenizer(self, args):
        if args.tokenizer == 'char':
            return tokenize_char
        elif args.tokenizer == 'word':
            return tokenize_word
        else:
            return tokenize_subword

    def _pad(self, vec, x):
        return np.pad(vec, (0, x), 'constant', constant_values=self.pad_idx)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        if i not in self._cache:
            entry = self.df.iloc[i]
            text, label = entry['text'], entry[self._label]
            tokens = _process_tokens(self.tokenizer(text))[:self.msl]
            if self.use_bert_encoder:
                token_ids = bert_tokenizer.encode(
                    tokens, add_special_tokens=True
                )
            else:
                token_ids = [self.enc.get(x, self.unk_idx) for x in tokens]
            token_ids = self._pad(token_ids, self.msl - len(tokens))
            x = torch.from_numpy(token_ids).long().to(self.device)
            y = torch.tensor(label).long().to(self.device)
            sl = torch.tensor(len(x)).float().to(self.device)
            self._cache[i] = (x, y, sl)
        return self._cache[i]


def create_loader(args, ds, shuffle=False):
    return DataLoader(ds, args.batch_size, shuffle)
