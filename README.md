# Detecting Perceived Emotions in Hurricane Disasters

Code and datasets for our ACL 2020 paper [Detecting Perceived Emotions in Hurricane Disasters](). If you found this project helpful, please consider citing our paper:

```bibtex
@inproceedings{desai-etal-2020-detecting,
  author={Desai, Shrey and Caragea, Cornelia and Li, Junyi Jessy},
  title={{Detecting Perceived Emotions in Hurricane Disasters}},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL)},
  year={2020},
}
```

## Abstract

Natural disasters (e.g., hurricanes) affect millions of people each year, causing widespread destruction in their wake. People have recently taken to social media websites (e.g., Twitter) to share their sentiments and feelings with the larger community. Consequently, these platforms have become instrumental in understanding and perceiving emotions at scale. In this paper, we introduce HurricaneEmo, an emotion dataset of 15,000 English tweets spanning three hurricanes: Harvey, Irma, and Maria. We present a comprehensive study of fine-grained emotions and propose classification tasks to discriminate between coarse-grained emotion groups. Our best BERT model, even after task-guided pre-training which leverages unlabeled Twitter data, achieves only 68% (averaged) accuracy. HurricaneEmo serves not only has a challenging benchmark for models but also as a valuable resource for analyzing emotions in disaster-centric domains.

## HurricaneEmo

We provide two versions of HurricaneEmo: binary and raw. The binary datasets have train, validation, and test splits for the coarse-grained Plutchik-8 binary classification task. The raw datasets are grouped by hurricane and have MTurk annotations for the fine-grained Plutchik-24 emotions; this can be useful for studying the full emotion spectrum.

### Binary Datasets

The splits for each emotion are shown below:

| Emotion        |  Train | Valid |  Test |
|----------------|:------:|:-----:|:-----:|
| aggressiveness |  4,209 |  526  |  527  |
| optimism       | 11,902 | 1,488 | 1,488 |
| love           |  2,569 |  321  |  322  |
| submission     |  6,092 |  762  |  762  |
| awe            |  7,324 |  916  |  916  |
| disapproval    |  5,931 |  741  |  742  |
| remorse        |  7,732 |  967  |  967  |
| contempt       |  3,763 |  470  |  471  |

Each Plutchik-8 emotion (aggressiveness, optimism, love, submission, awe, disapproval, remorse, contempt) has a train, validation, and test file. For example, the files for "awe" include `awe_train.csv`, `awe_valid.csv`, and `awe_test.csv`. Each file is a CSV containing of two headers---"text" and "\<emotion\>"---where \<emotion\> stands for one of the Plutchik-8 emotions. The \<emotion\> category has binary labels (0 or 1) for whether the tweet expresses that emotion or not.

```
$ head -n 10 awe_test.csv
text,awe
"as @tsfnetworks said , the # left will try to politicize # harvey to attack trump . pathetic .",0
houston ready ! thank you to everyone that has donated to kw cares : hurricane harvey victims . you have helped us hel https://t.co/b5fmqbxphs,1
groups of # volunteers are rushing to help those affected by # maria https://t.co/00foht5val https://t.co/uuculnwlml,0
@markhoppus @aspca @fullheadclub @markhoppus wrote a song for hurricane harvey dogs ! !,0
irma weakens to category 1 click below for full story ... https://t.co/6164xvjspr,0
a credible org to donate to our fellow americans in # puertorico # hurricanemaria # puertoricorelief,0
urban cowboy - tomorrow night ! - $ 1 from each ticket will go to hurricane harvey relief fund . tickets $ 5 . got ta l https://t.co/sa3fd2jrhi,0
"new "" red alert "" android banking trojan emerges https://t.co/tmstbsicyv # mobilesecurity # newsindustry # malware",0
hurricane irma destroys naples florida footage 9/10/2017 https://t.co/fnvkira4h7,0
```

### Raw Datasets

We also provide raw annotations for each hurricane: `harvey.jsonl`, `irma.jsonl`, and `maria.jsonl`. Each contains 5,000 tweets, which represents the number of tweets per hurricane we annotated. These files include fine-grained Plutchik-24 emotions for each tweet as provided by 5 annotators.

As an example, here is a sample of annotator #2's decisions on a Hurricane Harvey tweet:

```
{
  "text": "Brazoria County call center is open 979-864-1064. Thank you to our night shift  crew for taking care of out residents questions. #Harvey",
  "annotations": {
	...,
    "annotator2": {
      "acceptance": false,
      "admiration": true,
      "amazement": false,
      "anger": false,
      "annoyance": false,
      "anticipation": false,
      "apprehension": false,
      "boredom": false,
      "disgust": false,
      "distraction": false,
      "ecstasy": false,
      "fear": false,
      "grief": false,
      "interest": false,
      "joy": false,
      "loathing": false,
      "pensiveness": false,
      "rage": false,
      "sadness": false,
      "serenity": false,
      "surprise": false,
      "terror": false,
      "trust": false,
      "vigilance": true
    },
    ...,
  }
}
```

## Training Baselines

We provide the following commands to train the baseline models. These commands will train on `awe`, but you may replace this with any one of the Plutchik-8 emotions outlined above. Our models are trained on an NVIDIA Titan V GPU, and the results in the paper are an average of 10 runs with random restarts. If you are using non-pre-trained models, you will need to download the Twitter GloVe embeddings -- you may download our [pickled file](https://drive.google.com/file/d/1Wuu-F-mFd-Qjct1iUoxHh1wihXHRzYxd/view?usp=sharing) and place it in the root directory. Use the `--verbose` to see system output.

### Requirements

Our repository requires Python3.6+ and has the following dependencies:

 - `torch`
 - `numpy`
 - `sklearn`
 - `spacy` + `python3 -m spacy download en`
 - `transformers`

### Commands

**Logistic Regression**

```bash
$ python3 train.py \
	--model "lr"
	--ckpt "ckpt/lr.pt"
	--ds "datasets_binary/awe" \
	--tokenizer "word" \
	--batch_size 64 \
	--epochs 5 \
	--lr 1e-4
```

**Char CNN**

```bash
$ python3 train.py \
	--model "cnn" \
	--ckpt "ckpt/char_cnn.pt" \
	--ds "datasets_binary/awe" \
	--tokenizer "char" \
	--batch_size 64 \
	--epochs 5 \
	--lr 1e-3 \
	--ksizes 5 6 7 \
	--dropout 0.5
```

**Word CNN**

```bash
$ python3 train.py \
	--model "cnn" \
	--ckpt "ckpt/word_cnn.pt" \
	--ds "datasets_binary/awe" \
	--tokenizer "word" \
	--batch_size 64 \
	--epochs 5 \
	--lr 5e-5 \
	--ksizes 3 4 5 \
	--dropout 0.7
```

**GRU**

```bash
$ python3 train.py \
	--model "gru" \
	--ckpt "ckpt/gru.pt" \
	--ds "datasets_binary/awe" \
	--tokenizer "word" \
	--batch_size 64 \
	--epochs 5 \
	--lr 1e-4 \
	--dropout 0.7
```

**BERT**

```bash
$ python3 train.py \
	--model "bert" \
	--ckpt "ckpt/bert.pt" \
	--ds "datasets_binary/awe" \
	--tokenizer "bert" \
	--bert_model "bert-base-uncased" \
	--batch_size 8 \
	--grad_step 2 \
	--epochs 3 \
	--lr 2e-5 \
	--wd 0
```

**RoBERTa**

```bash
$ python3 train.py \
	--model "roberta" \
	--ckpt "ckpt/roberta.pt" \
	--ds "datasets_binary/awe" \
	--tokenizer "roberta" \
	--bert_model "roberta-base" \
	--batch_size 8 \
	--grad_step 2 \
	--epochs 3 \
	--lr 2e-5 \
	--wd 1e-3 \
	--pad_idx 1 \
	--unk_idx 3
```
