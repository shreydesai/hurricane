import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report

from main import args
from data import HurricaneDataset, create_loader
from models import LR, CNN, GRU, BertMLP, RobertaMLP


def select_model(args):
    model = None
    if args.model == 'lr':
        model = LR
    elif args.model == 'cnn':
        model = CNN
    elif args.model == 'gru':
        model = GRU
    elif args.model == 'bert':
        model = BertMLP
    elif args.model == 'roberta':
        model = RobertaMLP
    else:
        raise NotImplementedError
    return model(args).to(args.device)


def _grad_step(args, i):
    return args.grad_step == -1 or i % args.grad_step == 0


def train(args):
    model.train()
    train_loader = create_loader(args, train_ds)
    train_loss = 0.
    global_steps = 0
    optimizer.zero_grad()
    for i, (x, y, sl) in enumerate(train_loader, 1):
        loss = criterion(model(x, sl), y)
        if args.grad_step != -1:
            loss = loss / args.grad_step
        train_loss += loss.item()
        loss.backward()
        if _grad_step(args, i):
            optimizer.step()
            optimizer.zero_grad()
            global_steps += 1
    return train_loss / global_steps


def valid(args):
    model.eval()
    valid_loader = create_loader(args, valid_ds)
    valid_loss = 0.
    for (x, y, sl) in valid_loader:
        with torch.no_grad():
            logits = model(x, sl)
            loss = criterion(logits, y)
        valid_loss += loss.item()
    return valid_loss / len(valid_loader)


def test(args):
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    test_loader = create_loader(args, test_ds, shuffle=False)
    y_true, y_pred = [], []
    for (x, y, sl) in test_loader:
        with torch.no_grad():
            logits = model(x, sl)  # [B, C]
            preds = logits.argmax(dim=1)
        for i in range(len(x)):
            y_true.append(y[i].item())
            y_pred.append(preds[i].item())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return classification_report(y_true, y_pred, digits=4)


if args.verbose:
    print(args)
    print()

# Setup datasets.
train_ds = HurricaneDataset(args, 'train')
valid_ds = HurricaneDataset(args, 'valid')
test_ds = HurricaneDataset(args, 'test')

# Setup vocab.
if args.model == 'bert':
    args.n_emb = 30522
elif args.model == 'roberta':
    args.n_emb = 50265
else:
    args.n_emb = len(train_ds.enc)

# Setup training.
model = select_model(args)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.wd
)
best_loss = float('inf')
vl_history = []

# Setup pre-trained weights.
if args.model == 'bert' and args.pretrained_ckpt != '':
    prev_state_dict = torch.load(args.pretrained_ckpt, map_location='cpu')
    for n, p in model.named_parameters():
        if (
            n in prev_state_dict
            and n != 'fc.weight'
            and n != 'fc.bias'
        ):
        w = prev_state_dict[n]
        p.data.copy_(w.data)
    model = model.to(args.device)

# Setup embeddings (optional).
if args.model != 'bert' and args.model != 'roberta':
    glove = pickle.load(open('twitter_embs.pkl', 'rb'))
    embs = model.embedding.weight.clone()
    found = 0
    for i, word in enumerate(train_ds.vocab):
        if word in glove:
            embs[i] = torch.tensor(glove[word])
            found += 1
    if args.verbose:
        print(f'found {found}/{len(train_ds.vocab)} words')
    model.embedding.weight.data.copy_(embs)
    model = model.to(args.device)  # re-cuda

for epoch in range(1, args.epochs + 1):
    tl = train(args)
    vl = valid(args)
    vl_history.append(vl < best_loss)
    if vl < best_loss:
        best_loss = vl
        torch.save(model.state_dict(), args.ckpt)
    if args.verbose:
        print(
            f'epoch: {epoch} | '
            f'train loss: {tl:.6f} | '
            f'valid loss: {vl:.6f} | '
            f"{'*' if vl_history[-1] else ''}"
        )

report = test(args)

print()
print(report)
