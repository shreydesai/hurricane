import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--uid', type=str, default='')

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--ckpt', type=str)
parser.add_argument('--pretrained_ckpt', type=str, default='')
parser.add_argument('--verbose', action='store_true', default=False)

parser.add_argument('--pad_idx', type=int, default=0)
parser.add_argument('--unk_idx', type=int, default=1)

parser.add_argument('--ds', type=str)
parser.add_argument('--tokenizer', type=str, default='word')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--grad_step', type=int, default=-1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--max_seq_len', type=int, default=100)
parser.add_argument('--n_cls', type=int, default=2)
parser.add_argument('--model', type=str)

parser.add_argument('--n_emb', type=int, default=-1)  # placeholder
parser.add_argument('--emb_dim', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--n_chan', type=int, default=100)
parser.add_argument('--ksizes', nargs='+', type=int, default=[3, 4, 5])
parser.add_argument('--n_gru_hid', type=int, default=100)
parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
parser.add_argument('--n_bert_hid', type=int, default=768)

parser.add_argument('--ds_tag', type=str)
parser.add_argument('--train_path', type=str)
parser.add_argument('--valid_path', type=str)
parser.add_argument('--test_path', type=str)

parser.add_argument('--iters', type=int, default=10)

args = parser.parse_args()

if args.ds:
    args.ds_tag = f"{args.ds.split('/')[-1]}"  # datasets/awe -> awe
    args.train_path = f'{args.ds}_train.csv'  # datasets/awe_train.csv
    args.valid_path = f'{args.ds}_valid.csv'  # datasets/awe_valid.csv
    args.test_path = f'{args.ds}_test.csv'  # datasets/awe_test.csv
