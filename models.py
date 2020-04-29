import torch
import torch.nn as nn

from transformers import BertModel, RobertaModel

from main import args


def max_pool(x):
    return x.max(2)[0]


def mean_pool(x, sl):
    return torch.sum(x, 1) / sl.unsqueeze(1).float()


class LR(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding = nn.Embedding(args.n_emb, args.emb_dim)
        self.fc = nn.Linear(args.emb_dim, args.n_cls)

    def forward(self, x, sl):
       emb = self.embedding(x)  # [B, L, H_e]
       rep = mean_pool(emb, sl)  # [B, H_e]
       logits = self.fc(rep)
       return logits


class CNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.device = args.device

        self.embedding = nn.Embedding(args.n_emb, args.emb_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(args.emb_dim, args.n_chan, k, padding=k)
                for k in args.ksizes
            ]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(
            args.n_chan * len(args.ksizes), args.n_cls
        )

    def conv_and_pool(self, x, conv):
        return max_pool(self.relu(conv(x)))

    def forward(self, x, sl=None):
        emb = self.embedding(x).transpose(1, 2)
        rep = [self.conv_and_pool(emb, conv) for conv in self.convs]
        rep = torch.cat(rep, dim=1).to(self.device)
        logits = self.fc(self.dropout(rep))
        return logits


class GRU(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding = nn.Embedding(args.n_emb, args.emb_dim)
        self.gru = nn.GRU(args.emb_dim, args.n_gru_hid)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.n_gru_hid, args.n_cls)

    def forward(self, x, sl):
        emb = self.embedding(x)  # [B, L, H_e]
        out, _ = self.gru(emb)  # [B, L, H_l]
        rep = mean_pool(out, sl)  # [B, H_l]
        logits = self.fc(self.dropout(rep))  # [B, C]
        return logits


class BertMLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bert = BertModel.from_pretrained(args.bert_model)
        self.fc = nn.Linear(args.n_bert_hid, args.n_cls)

    def forward(self, x, sl=None):
        mask = (x != 0).float()
        emb, _ = self.bert(x, attention_mask=mask)  # [B, L, H_b]
        rep = emb[:, 0, :]  # [B, H_b]
        logits = self.fc(rep)  # [B, C]
        return logits


class RobertaMLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(args.bert_model)
        self.fc = nn.Linear(args.n_bert_hid, args.n_cls)

    def forward(self, x, sl=None):
        mask = (x != 1).float()
        emb, _ = self.roberta(x, attention_mask=mask)  # [B, L, H_b]
        rep = emb[:, 0, :]  # [B, H_b]
        logits = self.fc(rep)  # [B, C]
        return logits
