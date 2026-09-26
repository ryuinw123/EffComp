#!/usr/bin/env python
# coding: utf-8


import argparse
import math
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain the EffComp sentence selection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    settings = {
        "dataset": ("document_hotpotqa_train.json", "Input JSON array (.json or .txt)"),
        "model": ("answerdotai/ModernBERT-base", "Base model and tokenizer path or identifier"),
        "output_dir": ("test", "Training checkpoint output directory"),
        "max_length": (8192, "Maximum input length in tokens"),
        "sentence_token_id": (50286, "Sentence marker token ID"),
        "num_proc": (4, "Dataset preprocessing workers"),
        "validation_ratio": (0.1, "Fraction of examples reserved for validation"),
        "seed": (42, "Training and dataset split seed"),
        "learning_rate": (2e-5, "Optimizer learning rate"),
        "epochs": (2, "Number of training epochs"),
        "weight_decay": (0.01, "Optimizer weight decay"),
        "train_batch_size": (4, "Training batch size per device"),
        "eval_batch_size": (4, "Evaluation batch size per device"),
        "cuda_visible_devices": (os.environ.get("CUDA_VISIBLE_DEVICES", "0"), "Visible CUDA devices"),
    }
    for name, (default, help_text) in settings.items():
        flags = ["--" + name.replace("_", "-")]
        if "_" in name:
            flags.append("--" + name)
        parser.add_argument(*flags, dest=name, type=type(default), default=default, help=help_text)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="Enable bfloat16 training")
    args = parser.parse_args()
    for name in ("max_length", "num_proc", "learning_rate", "epochs", "train_batch_size", "eval_batch_size"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not 0 < args.validation_ratio < 1:
        parser.error("--validation-ratio must be between 0 and 1 (exclusive)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative and finite")
    if args.seed < 0 or args.sentence_token_id < 0:
        parser.error("--seed and --sentence-token-id must be non-negative")
    return args


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
from transformers import AutoModelForSequenceClassification, AutoTokenizer , Trainer , TrainingArguments , AutoModelForTokenClassification , AutoModel
from sklearn.utils.class_weight import compute_class_weight
import evaluate
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize


import collections
import logging
import regex
import string
import unicodedata
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import List
import numpy as np

from collections import Counter

"""
Evaluation code from DPR: https://github.com/facebookresearch/DPR
"""

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

logger = logging.getLogger(__name__)

QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits'])

def calculate_matches(data: List, workers_num: int):
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """

    logger.info('Matching answers in top docs...')

    tokenizer = SimpleTokenizer()
    get_score_partial = partial(check_answer, tokenizer=tokenizer)

    processes = ProcessPool(processes=workers_num)
    scores = processes.map(get_score_partial, data)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(data[0]['ctxs'])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)

def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for i, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits

def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

#################################################
########        READER EVALUATION        ########
#################################################

def _normalize(text):
    return unicodedata.normalize('NFD', text)

#Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

####################################################
########        RETRIEVER EVALUATION        ########
####################################################

def eval_batch(scores, inversions, avg_topk, idx_topk):
    for k, s in enumerate(scores):
        s = s.cpu().numpy()
        sorted_idx = np.argsort(-s)
        score(sorted_idx, inversions, avg_topk, idx_topk)

def count_inversions(arr):
    inv_count = 0
    lenarr = len(arr)
    for i in range(lenarr):
        for j in range(i + 1, lenarr):
            if (arr[i] > arr[j]):
                inv_count += 1
    return inv_count

def score(x, inversions, avg_topk, idx_topk):
    x = np.array(x)
    inversions.append(count_inversions(x))
    for k in avg_topk:
        # ratio of passages in the predicted top-k that are
        # also in the topk given by gold score
        avg_pred_topk = (x[:k]<k).mean()
        avg_topk[k].append(avg_pred_topk)
    for k in idx_topk:
        below_k = (x<k)
        # number of passages required to obtain all passages from gold top-k
        idx_gold_topk = len(x) - np.argmax(below_k[::-1])
        idx_topk[k].append(idx_gold_topk)

"""
our evaluation code
"""

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    ZERO_METRIC = (0, 0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    accuracy = 1.0 if normalized_ground_truth in normalized_prediction else 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, precision, recall, accuracy)

def f1_scores(prediction, ground_truths):
    max_f1 = max_precision = max_recall = max_acc = 0

    for gt in ground_truths:
        score = f1_score(prediction, gt)
        max_f1 = max(max_f1, score[0])  # Accessing F1 score from the tuple
        max_precision = max(max_precision, score[1])  # Accessing precision from the tuple
        max_recall = max(max_recall, score[2])  # Accessing recall from the tuple
        max_acc = max(max_acc, score[3]) 
    
    return max_f1, max_precision, max_recall, max_acc

def evaluate_QA(results, ans_key, predict_key):
    """
    EVALUATION
    """
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0, 'acc': 0}    
            
    em_for_task = ems
    f1_for_task = f1_scores
        
    for result in results:
        prediction = result[predict_key]
        gold = result[ans_key]
    
        em = em_for_task(prediction, gold)
        f1, prec, recall, acc = f1_for_task(prediction, gold)
        
        metrics['em'] += float(em)
        metrics['f1'] += f1
        metrics['prec'] += prec
        metrics['recall'] += recall
        metrics['acc'] += acc
                
        result['metrics'] = {'em': float(em), 'f1': f1, 'prec': prec, 'recall': recall, 'acc': acc}

    for k in metrics.keys():
        metrics[k] /= len(results)
    
    return metrics


import ijson
from datasets import Dataset
from decimal import Decimal


def safe_convert(value):
    if isinstance(value, Decimal):
        return float(value)  # or str(value) if you want string format
    elif isinstance(value, dict):
        return {k: safe_convert(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [safe_convert(v) for v in value]
    else:
        return value

def stream_json_array(file_path):
    with open(file_path, 'rb') as f:
        for item in ijson.items(f, 'item'):
            yield safe_convert(item)

dataset = Dataset.from_generator(lambda: stream_json_array(args.dataset))


sentence_token_id = args.sentence_token_id
ignore_token_id = -100


model_checkpoint = args.model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
simple_tokenizer = SimpleTokenizer()


max_length = args.max_length
def training_filter(example):
    question = example["question"]
    ctxs = example["ctxs"]

    answers = example["answers"]
    if any(normalize_answer(answer) in ['yes', 'no', 'noanswer'] for answer in answers):
        return False

    context = "\n".join(ctx["text"] for ctx in ctxs)
    split_context = sent_tokenize(context)
    context_input_ids = tokenizer(split_context , add_special_tokens = False , return_attention_mask = False)["input_ids"]

    question_input_ids = tokenizer(question , return_attention_mask = False)
    question_ids = question_input_ids["input_ids"]


    all_input_ids = question_ids

    for segment_input_ids in context_input_ids:
        all_input_ids.append(sentence_token_id)
        segment_len = len(segment_input_ids)
        all_input_ids.extend(segment_input_ids)
    all_input_ids.append(tokenizer.sep_token_id)

    if len(all_input_ids) > max_length:
        return False
    return True


dataset


train_dataset = dataset.filter(
    training_filter,
    num_proc = args.num_proc
)


train_dataset


train_dataset = train_dataset.train_test_split(test_size=args.validation_ratio, seed=args.seed)


train_dataset


train_dataset


def preprocess_training_examples(example):
    question = example["question"]
    ctxs = example["ctxs"]
    answers = example["answers"]


    context = "\n".join(ctx["text"] for ctx in ctxs)
    splits_context = sent_tokenize(context)
    context_input_ids = tokenizer(splits_context , add_special_tokens = False , return_attention_mask = False)["input_ids"]
    contexts_has_answers = [
        1 if has_answer(answers, context, simple_tokenizer) else 0
        for context in splits_context
    ]
    
    

    question_input_ids = tokenizer(question , return_attention_mask = False)
    question_ids = question_input_ids["input_ids"]


    all_input_ids = question_ids
    labels = [ignore_token_id] * (len(question_ids))

    for segment_input_ids , context_has_answer in zip(context_input_ids , contexts_has_answers):
        all_input_ids.append(sentence_token_id)
        labels.append(context_has_answer)
        segment_len = len(segment_input_ids)
        all_input_ids.extend(segment_input_ids)
        labels.extend([ignore_token_id] * segment_len)
        
    all_input_ids.append(tokenizer.sep_token_id)
    labels.append(ignore_token_id)

    inputs = {
        "input_ids" : all_input_ids
    }

    input_len = max_length - len(all_input_ids)
    labels.extend([ignore_token_id] * input_len)

    inputs = tokenizer.pad(inputs, padding = "max_length", max_length = max_length)
    inputs["labels"] = labels


    assert len(labels) == len(inputs["input_ids"])

    return inputs


train_dataset = train_dataset.map(
    preprocess_training_examples,
    remove_columns=dataset.column_names,
    
    num_proc = args.num_proc
)


label_false = 0
label_true = 0


for data in train_dataset["train"]:
    labels = list(filter(lambda a: a != ignore_token_id, data["labels"]))
    label_false += (len(labels) - sum(labels))  # count of 0s
    label_true += sum(labels)                 # count of 1s


label_true


all_label = [0] * label_false + [1] * label_true
ce_weight = compute_class_weight(class_weight="balanced", classes=np.array([0 , 1]), y=all_label)


ce_weight


def compute_metrics(eval_preds):
    metric = evaluate.load("recall")
    logits, labels = eval_preds
    # Get predicted class for each token
    predictions = np.argmax(logits, axis=-1)

    # Flatten predictions and labels
    pred_flat = predictions.flatten()
    labels_flat = labels.flatten()

    mask = labels_flat != ignore_token_id

    filtered_preds = pred_flat[mask]
    filtered_labels = labels_flat[mask]
    
    return metric.compute(predictions=filtered_preds, references=filtered_labels)


class CompressorModel(nn.Module):
    def __init__(self, checkpoint, ce_weight=None):
        super(CompressorModel, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint , num_labels=2)
        
        # Register weights if provided (shape: [num_classes])
        if ce_weight is not None:
            self.register_buffer("ce_weight", torch.tensor(ce_weight, dtype=torch.float))
        else:
            self.ce_weight = None

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                weight=self.ce_weight,
                ignore_index=ignore_token_id,
            )

        return {
            "loss": loss,
            "logits": logits,
        }


model = CompressorModel(model_checkpoint , ce_weight)


training_args = TrainingArguments(
    args.output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    bf16=args.bf16,
    seed=args.seed,)


trainer = Trainer(
    model,
    training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=train_dataset["test"],
    processing_class =tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()


