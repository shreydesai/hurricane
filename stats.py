import argparse
import collections
import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
args = parser.parse_args()

data = json.load(open(args.path))

workers = set()
sent_lens = []
mentions = []
links = []
vocab = collections.Counter()
vocab_filtered = collections.Counter()

for tweet in data:
    text = tweet['text']
    tokens = [x.lower() for x in text.split()]
    tokens_filtered = [
        x for x in tokens
        if not x.startswith('@') and not x.startswith('http')
    ]
    mention = sum([1 if x.startswith('@') else 0 for x in tokens])
    link = sum([1 if x.startswith('http') else 0 for x in tokens])
    for resp in tweet['resps']:
        workers.add(resp)
    sent_lens.append(len(tokens))
    vocab.update(tokens)
    vocab_filtered.update(tokens_filtered)
    mentions.append(1 if mention > 0 else 0)
    links.append(1 if link > 0 else 0)

print(f'workers: {len(workers)}')
print(f'sent len (mean): {np.mean(sent_lens):.4f}')
print(f'sent len (median): {np.median(sent_lens):.4f}')
print(f'vocab (total): {len(vocab)}')
print(f'vocab (filtered): {len(vocab_filtered)}')
print(f'mentions (%): {np.mean(mentions):.4f}')
print(f'links (%): {np.mean(links):.4f}')