#!/usr/bin/env python
# coding: utf-8


import argparse
import math


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and evaluate the EffComp sentence compression policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    settings = {
        "rollout_size": (128, "Number of samples per policy rollout"),
        "lr": (1e-6, "Optimizer learning rate"),
        "epoch": (15, "Total number of training epochs, including resumed epochs"),
        "filter_device": ("cuda", "Device for the compression policy"),
        "generator_device": ("cuda", "Device for the answer generator"),
        "filter_model": ("answerdotai/ModernBERT-base", "Compression model path or Hugging Face identifier"),
        "gen_model": ("google/gemma-2-9b-it", "Answer generator path or Hugging Face identifier"),
        "seed": (42, "Dataset split seed"),
        "max_length": (2048, "Maximum input length in tokens"),
        "sentence_token_id": (50286, "Token ID marking the start of each context sentence"),
        "dataset": ("bge-hotpotqa", "Dataset directory saved with save_to_disk"),
        "inference_batch_size": (8, "Policy inference batch size (at least 2)"),
        "backward_batch_size": (8, "Policy optimization batch size"),
        "n_updates_per_iteration": (4, "Policy optimization passes per rollout"),
        "enf_coefficient": (0.1, "Entropy regularization coefficient"),
        "clip_value": (0.2, "Policy probability ratio clipping range"),
        "max_grad_norm": (0.5, "Maximum gradient norm"),
        "log_path": ("eval_log.csv", "Evaluation CSV output path"),
        "save_pattern": ("agent_best_filter_epoch_*.pth", "Glob used to find checkpoints to resume; does not change save filenames"),
        "alpha": (0.95, "Accuracy reward weight between 0 and 1"),
        "target_kl": (0.02, "KL divergence threshold for early stopping"),
        "llm_batch_size": (8, "Answer generator batch size"),
        "validation_size": (1000, "Number of examples reserved for validation"),
    }
    for name, (default, help_text) in settings.items():
        flags = ["--" + name.replace("_", "-")]
        if "_" in name:
            flags.append("--" + name)
        parser.add_argument(*flags, dest=name, type=type(default), default=default, help=help_text)
    for name, default, description in (
        ("use_replay", True, "Enable replay training"),
        ("is_gold", False, "Prefer contexts containing gold answers"),
    ):
        group = parser.add_mutually_exclusive_group()
        flag = name.replace("_", "-")
        group.add_argument("--" + flag, "--" + name, dest=name, action="store_true", help=description)
        group.add_argument("--no-" + flag, dest=name, action="store_false", help="Disable: " + description.lower())
        parser.set_defaults(**{name: default})

    args = parser.parse_args(argv)
    for name in ("rollout_size", "lr", "epoch", "max_length", "inference_batch_size",
                 "backward_batch_size", "n_updates_per_iteration", "clip_value",
                 "max_grad_norm", "target_kl", "llm_batch_size", "validation_size"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive, finite number")
    if args.inference_batch_size < 2:
        parser.error("--inference-batch-size must be at least 2")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    if not math.isfinite(args.enf_coefficient) or args.enf_coefficient < 0:
        parser.error("--enf-coefficient must be a non-negative, finite number")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.sentence_token_id < 0:
        parser.error("--sentence-token-id must be non-negative")
    return args

# Parse before importing ML dependencies so --help is available without them.
hyperparameters = vars(parse_args())


def print_metrics(stage, reward, accuracy, compression_rate):
    print(
        f"{stage} | Mean reward: {reward:.4f} | "
        f"Accuracy: {accuracy:.2%} | Compression rate: {compression_rate:.2%}",
        flush=True,
    )


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from datasets import load_from_disk , DatasetDict , Dataset
from tqdm import tqdm
from transformers import AutoTokenizer , AutoModel , pipeline , AutoModelForTokenClassification , BitsAndBytesConfig , AutoModelForCausalLM
from torch.optim import AdamW
from torch.utils.data import DataLoader
import glob
torch.set_printoptions(threshold=float('inf'))
from nltk.tokenize import sent_tokenize
import csv
from sklearn.utils.class_weight import compute_class_weight
import time

from transformers.utils import logging
logging.set_verbosity_error()


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


class Env():
    def __init__(self , model_path , device , alpha , batch_size = 8 , max_length = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="eager",
            device_map = device
        )
        self.pipeline = pipeline(
            "text-generation",
             model=model,
             tokenizer = self.tokenizer,
             batch_size=batch_size,
             model_max_length = max_length
        )
        self.batch_size = batch_size
        self.alpha = alpha
    
    def generate_batch_predictions(self,batch):
        predictions = []
        for i in range(0, len(batch), self.batch_size):
            start = i
            end = start + self.batch_size
            sub_batch = batch[start:end]
            chat_batch = [
                    [{"role": "user", "content": f"""You are an expert in Question Answering. Your job is to answer questions in 1 to 5 words based on the given context. Question: {item['question']}\nContext: {item['context']}\nAnswer:"""}]
                    for item in sub_batch
                ]
    
            outputs = self.pipeline(
                chat_batch,
                return_full_text=False,
                max_new_tokens=50,
                do_sample=False
            )
            predictions.extend([
                output[0]["generated_text"].removeprefix("assistant").strip()
                for output in outputs
            ])
        return predictions

    def step(self , input_ids,actions,sentence_action_mask,sentence_mask,questions,answers):
        # next_obs, reward, done, info = envs.step(action.cpu().numpy())
        llm_batch = []
        compression_rates = []
        rewards = []
        results = []
        assert len(input_ids) == len(sentence_action_mask) == len(questions) == len(answers)
        for ids , act , sa_mask , s_mask , question in zip(input_ids,actions,sentence_action_mask,sentence_mask,questions):

            sentence_selected = s_mask[(sa_mask == 1) & (act == 1)]
            mask = torch.isin(s_mask, sentence_selected) & (sa_mask != 1)
            kept_ids = ids[mask]

            llm_batch.append({"context" : tokenizer.decode(kept_ids) , "question" : question})

            total_tokens = ((s_mask != ignore_token_id) & (sa_mask != 1)).sum().item()
            kept_tokens = len(kept_ids)
            eps = np.finfo(np.float32).eps.item()
            compression_rate = 1-(kept_tokens / (total_tokens + eps))
            compression_rates.append(compression_rate)

        llm_response = self.generate_batch_predictions(llm_batch)

        for response , answer , compression_rate in zip(llm_response , answers , compression_rates):
            match_result = f1_scores(response , answer)[3]
            is_correct = int(match_result)
            compression_reward = compression_rate if is_correct else -compression_rate
            reward = self.alpha * is_correct + (1 - self.alpha) * compression_reward
            rewards.append(reward)
            results.append({
                "prediction": response,
                "gold": answer,
            })

        metrics = evaluate_QA(results, "gold", "prediction")

        info = {
            "mean_compress_rate" : sum(compression_rates) / len(compression_rates),
            "mean_rewards" : sum(rewards) / len(rewards),
            "f1" : metrics["f1"],
            "em" : metrics["em"],
            "accuracy" : metrics["acc"]
        }
        
        return rewards , info


class Agent(nn.Module):
    def __init__(self , filter_model , device):
        super().__init__()
        self.base_model = AutoModelForTokenClassification.from_pretrained(filter_model , num_labels=2).to(device)
        self.device = device
        
    def forward(self , input_ids , attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask , output_hidden_states = True)
        return outputs
        
    @torch.no_grad()    
    def get_action(self, input_ids , attention_mask , sentence_action_mask, deterministic = False):
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = Categorical(logits=logits)
        
        if (deterministic):
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = probs.sample()

        log_prob = probs.log_prob(actions)
        masked_log_prob = torch.sum(log_prob * sentence_action_mask , dim = -1) / torch.sum(sentence_action_mask , dim = -1)
        
        return actions , masked_log_prob

    def evaluate(self, input_ids , attention_mask, actions , sentence_action_mask):
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = Categorical(logits=logits)
        log_prob = probs.log_prob(actions)
        masked_log_prob = torch.sum(log_prob * sentence_action_mask , dim = -1) / torch.sum(sentence_action_mask , dim = -1)
        entropy = probs.entropy()
        masked_entropy = torch.sum(entropy * sentence_action_mask , dim = -1) / torch.sum(sentence_action_mask , dim = -1)
        
        return masked_log_prob, masked_entropy


def shuffle_in_unison_(lists, generator=None):
    """
    In-place shuffle of multiple same-length lists using a single random permutation.
    Usage: shuffle_in_unison_([list1, list2, ...])
    """
    n = len(lists[0])
    assert all(len(x) == n for x in lists), "All lists must have the same length"

    idx = torch.randperm(n, generator=generator).tolist()
    for k, lst in enumerate(lists):
        lists[k][:] = [lst[i] for i in idx]  # mutate in-place


def learn(policy , optimizer , batch_input_ids , batch_attention_mask , batch_log_prob , batch_action , batch_sentence_action_mask , batch_reward , backward_batch_size , n_updates_per_iteration , enf_coefficient , clip_value , max_grad_norm , target_kl):
    for _ in range(n_updates_per_iteration):
        shuffle_in_unison_([
            batch_input_ids,
            batch_attention_mask,
            batch_log_prob,
            batch_action,
            batch_sentence_action_mask,
            batch_reward,
        ])
        for i in range(0, len(batch_input_ids), backward_batch_size):
            start = i
            end = i + backward_batch_size
            input_ids = torch.tensor(batch_input_ids[start:end]).to(policy.device)
            attention_mask = torch.tensor(batch_attention_mask[start:end]).to(policy.device)
            actions = torch.tensor(batch_action[start:end]).to(policy.device)
            sa_mask = torch.tensor(batch_sentence_action_mask[start:end]).to(policy.device)
            rewards = torch.tensor(batch_reward[start:end]).to(policy.device)
            prev_log_probs = torch.tensor(batch_log_prob[start:end]).to(policy.device)
            with torch.amp.autocast(policy.device,dtype=torch.bfloat16):
                curr_log_probs , entropy = policy.evaluate(input_ids , attention_mask , actions , sa_mask)
                logratios = curr_log_probs - prev_log_probs
                ratios = torch.exp(logratios)
                approx_kl = ((ratios - 1) - logratios).mean()
                surr1 = ratios * rewards
                surr2 = torch.clamp(ratios, 1 - clip_value, 1 + clip_value) * rewards
                policy_loss = (-torch.min(surr1, surr2)).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - (enf_coefficient * entropy_loss)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
        if target_kl is not None:
            if approx_kl > target_kl:
                break


class ReplayBuffer:
    def __init__(self , threshold):
        self.buffer = {}  # key: index, value: best sample (dict)
        self.threshold = threshold

    def insert(self, index, input_ids, attention_mask, sentence_action_mask, action ,reward , question , ans):

        if reward < self.threshold:
            return
        
        new_sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentence_action_mask": sentence_action_mask,
            "action": action,
            "reward": reward,
            "question" : question,
            "answers" : ans,
        }

        # Insert if no sample exists, or new reward > existing (replace on tie or better)
        if index not in self.buffer or reward > self.buffer[index]["reward"]:
            self.buffer[index] = new_sample

    def update(self, indexs, batch_input_ids, batch_attention_mask, batch_sentence_action_mask, batch_action, batch_reward , questions,answers):
        for index, input_ids, attention_mask, sa_mask, action, reward , question , ans in zip(
            indexs, batch_input_ids, batch_attention_mask, batch_sentence_action_mask, batch_action, batch_reward , questions , answers
        ):
            self.insert(index, input_ids, attention_mask, sa_mask, action, reward , question , ans)
    
    def export_dataset(self):
        untrained_items = {
            k: v for k, v in self.buffer.items()
        }

        if not untrained_items:
            return None

        buffer_list = [
            untrained_items[k]
            for k in sorted(untrained_items.keys())
        ]

        flat_dict = {
            key: [sample[key] for sample in buffer_list]
            for key in buffer_list[0]
        }

        dataset = Dataset.from_dict(flat_dict)

        return dataset

    def import_empty_dataset(self , dataset):
        for data in dataset:
            self.insert(data["index"] , data["input_ids"] , data["attention_mask"] , data["sentence_action_mask"] , [1] * len(data["sentence_action_mask"]) , self.threshold , data["question"] , data["answers"])
    
    def __len__(self):
        return len(self.buffer)

    def export_buffer(self):
        return self.buffer
        
    def import_buffer(self, buffer):
        self.buffer = buffer


def train(policy, env , optimizer, dataloader , rb , rollout_size , backward_batch_size , n_updates_per_iteration , enf_coefficient , clip_value , max_grad_norm , alpha , target_kl):

    policy.train()
    global_acc = []
    global_compress_rate = []
    global_reward = []
    

    
    batch_input_ids = []
    batch_attention_mask = []
    batch_sentence_action_mask = []
    batch_sentence_mask = []
    batch_question = []
    batch_answers = []
    batch_action = []
    batch_log_prob = []
    batch_reward = []
    batch_score = []

    batch_acc = []
    batch_compress_rate = []
    
    for batch in tqdm(dataloader, desc="dataset batches"):
        indexs = batch["indexs"]
        input_ids = batch["input_ids"].to(policy.device)
        attention_mask = batch["attention_mask"].to(policy.device)
        sentence_action_mask = batch["sentence_action_mask"].to(policy.device)
        sentence_mask = batch["sentence_mask"].to(policy.device)
        questions = batch["questions"]
        answers = batch["answers"]
        
        actions, log_probs = policy.get_action(input_ids , attention_mask , sentence_action_mask)
        rewards, info = env.step(input_ids,actions,sentence_action_mask,sentence_mask,questions,answers)

        input_ids = input_ids.tolist()
        attention_mask = attention_mask.tolist()
        sentence_action_mask = sentence_action_mask.tolist()
        actions = actions.tolist()
        log_probs = log_probs.tolist()

        if hyperparameters["use_replay"]:
            rb.update(indexs,input_ids,attention_mask,sentence_action_mask,actions,rewards,questions,answers)
            
        rewards = torch.tensor(rewards)     
        if rewards.max() != rewards.min():
            rewards = (rewards - rewards.mean()) / (rewards.std())
        else:
            rewards = torch.zeros_like(rewards)


        mean_accuracy_score = info["accuracy"]
        mean_compress_rate = info["mean_compress_rate"]
        mean_reward = info["mean_rewards"]

        batch_input_ids.extend(input_ids)
        batch_attention_mask.extend(attention_mask)
        batch_sentence_action_mask.extend(sentence_action_mask)
        batch_sentence_mask.extend(sentence_mask)
        batch_question.extend(questions)
        batch_answers.extend(answers)
        batch_action.extend(actions)
        batch_log_prob.extend(log_probs)
        batch_reward.extend(rewards)

        batch_acc.append(mean_accuracy_score)
        batch_compress_rate.append(mean_compress_rate)
        batch_score.append(mean_reward)

        global_acc.append(mean_accuracy_score)
        global_compress_rate.append(mean_compress_rate)
        global_reward.append(mean_reward)
        
        if len(batch_input_ids) >= rollout_size:
            learn(policy , optimizer , batch_input_ids , batch_attention_mask , batch_log_prob , batch_action , batch_sentence_action_mask , batch_reward , backward_batch_size , n_updates_per_iteration , enf_coefficient , clip_value , max_grad_norm , target_kl)
            avg_acc = sum(batch_acc) / len(batch_acc)
            avg_compress_rate = sum(batch_compress_rate) / len(batch_compress_rate)
            avg_rewards = sum(batch_score) / len(batch_score)
            print_metrics("Training rollout", avg_rewards, avg_acc, avg_compress_rate)
            batch_input_ids = []
            batch_attention_mask = []
            batch_sentence_action_mask = []
            batch_sentence_mask = []
            batch_question = []
            batch_answers = []
            batch_action = []
            batch_log_prob = []
            batch_reward = []
            batch_acc = []
            batch_compress_rate = []
            batch_score = []

    avg_acc = sum(global_acc) / len(global_acc)
    avg_compress_rate = sum(global_compress_rate) / len(global_compress_rate)
    avg_rewards = sum(global_reward) / len(global_reward)
    
    return avg_rewards , avg_acc , avg_compress_rate


def replay_cold_start(policy , env , dataset , rb , inference_batch_size):

    def collate_fn(batch):
        return {
                "indexs" : [b["index"] for b in batch],
                "input_ids": torch.tensor([b["input_ids"] for b in batch]),
                "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
                "sentence_mask": torch.tensor([b["sentence_mask"] for b in batch]),
                "sentence_action_mask": torch.tensor([b["sentence_action_mask"] for b in batch]),
                "questions": [b["question"] for b in batch],
                "answers": [b["answers"] for b in batch],
            }
    
    dataloader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, collate_fn=collate_fn)
    
    batch_rewards = []
    batch_acc = []
    batch_compress_rate = []
    
    for batch in tqdm(dataloader, desc="dataset batches"):
        indexs = batch["indexs"]
        input_ids = batch["input_ids"].to(policy.device)
        attention_mask = batch["attention_mask"].to(policy.device)
        sentence_action_mask = batch["sentence_action_mask"].to(policy.device)
        sentence_mask = batch["sentence_mask"].to(policy.device)
        questions = batch["questions"]
        answers = batch["answers"]
        
        actions, log_probs = policy.get_action(input_ids , attention_mask , sentence_action_mask , deterministic = True)
        rewards, info = env.step(input_ids,actions,sentence_action_mask,sentence_mask,questions,answers)

        input_ids = input_ids.tolist()
        attention_mask = attention_mask.tolist()
        sentence_action_mask = sentence_action_mask.tolist()
        actions = actions.tolist()
        log_probs = log_probs.tolist()

        rb.update(indexs,input_ids,attention_mask,sentence_action_mask,actions,rewards,questions,answers)

        acc_score = info["accuracy"]
        compress_rate = info["mean_compress_rate"]
        batch_rewards.extend(rewards)
        batch_acc.append(acc_score)
        batch_compress_rate.append(compress_rate)
        
    mean_reward = sum(batch_rewards) / len(batch_rewards)
    mean_acc = sum(batch_acc) / len(batch_acc)
    mean_compress_rate = sum(batch_compress_rate) / len(batch_compress_rate)
    return mean_reward , mean_acc , mean_compress_rate


def evaluate(policy , env , dataloader):
    policy.eval()
    batch_rewards = []
    batch_acc = []
    batch_compress_rate = []
    
    for batch in tqdm(dataloader, desc="dataset batches"):
        input_ids = batch["input_ids"].to(policy.device)
        attention_mask = batch["attention_mask"].to(policy.device)
        sentence_action_mask = batch["sentence_action_mask"].to(policy.device)
        sentence_mask = batch["sentence_mask"].to(policy.device)
        questions = batch["questions"]
        answers = batch["answers"]
        
        actions, log_probs = policy.get_action(input_ids , attention_mask , sentence_action_mask , deterministic = True)
        rewards, info = env.step(input_ids,actions,sentence_action_mask,sentence_mask,questions,answers)

        acc_score = info["accuracy"]
        compress_rate = info["mean_compress_rate"]
        batch_rewards.extend(rewards)
        batch_acc.append(acc_score)
        batch_compress_rate.append(compress_rate)
    mean_reward = sum(batch_rewards) / len(batch_rewards)
    mean_acc = sum(batch_acc) / len(batch_acc)
    mean_compress_rate = sum(batch_compress_rate) / len(batch_compress_rate)
    return mean_reward , mean_acc , mean_compress_rate


print("EffComp training configuration", flush=True)
for name, value in sorted(hyperparameters.items()):
    print(f"  {name}: {value}", flush=True)


assert hyperparameters["inference_batch_size"] >= 2


sentence_token_id = hyperparameters["sentence_token_id"]
ignore_token_id = -100


dataset = load_from_disk(hyperparameters["dataset"])


tokenizer = AutoTokenizer.from_pretrained(hyperparameters["filter_model"])
simple_tokenizer = SimpleTokenizer()


max_length = hyperparameters["max_length"]

if hyperparameters["is_gold"]:
    def filter_examples(example):
        question = example["question"]
        ctxs = example["ctxs"]
    
        answers = example["answers"]

        all_context = [ctx["text"] for ctx in ctxs if ctx["has_answer"]]
        if len(all_context) <= 0 or any(normalize_answer(answer) in ['yes', 'no', 'noanswer'] for answer in answers):
            ctxs = ctxs[:10]
            all_context = [ctx["text"] for ctx in ctxs]
        
        context = "\n".join(all_context)
    
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

    def preprocess(example):
        question = example["question"]
        ctxs = example["ctxs"]
        answers = example["answers"]
    
    
        all_context = [ctx["text"] for ctx in ctxs if ctx["has_answer"]]
        if len(all_context) <= 0 or any(normalize_answer(answer) in ['yes', 'no', 'noanswer'] for answer in answers):
            ctxs = ctxs[:10]
            all_context = [ctx["text"] for ctx in ctxs]
        
        context = "\n".join(all_context)
        
        splits_context = sent_tokenize(context)
        context_input_ids = tokenizer(splits_context , add_special_tokens = False , return_attention_mask = False)["input_ids"]
        
    
        question_input_ids = tokenizer(question , return_attention_mask = False)
        question_ids = question_input_ids["input_ids"]
    
    
        all_input_ids = question_ids
        sentence_mask = [ignore_token_id] * (len(question_ids))
        sentence_action_mask = [0] * (len(question_ids))
        sentence_count = 0
    
        for segment_input_ids in context_input_ids:
            all_input_ids.append(sentence_token_id)
            segment_len = len(segment_input_ids)
            all_input_ids.extend(segment_input_ids)
            sentence_action_mask.append(1)
            sentence_action_mask.extend([0] * segment_len)
            sentence_mask.append(sentence_count)
            sentence_mask.extend([sentence_count] * segment_len)
            sentence_count += 1
            
        all_input_ids.append(tokenizer.sep_token_id)
        sentence_action_mask.append(0)
        sentence_mask.append(ignore_token_id)
    
        inputs = {
            "input_ids" : all_input_ids
        }
    
        input_len = max_length - len(all_input_ids)
        sentence_action_mask.extend([0] * input_len)
        sentence_mask.extend([ignore_token_id] * input_len)
    
        inputs = tokenizer.pad(inputs, padding = "max_length", max_length = max_length)
        inputs["sentence_action_mask"] = sentence_action_mask
        inputs["sentence_mask"] = sentence_mask
        inputs["question"] = question
        inputs["answers"] = answers
    
        assert len(inputs["input_ids"]) == len(inputs["sentence_mask"]) == len(inputs["sentence_action_mask"])
    
        return inputs
else:
    def filter_examples(example):
        question = example["question"]
        ctxs = example["ctxs"][:10]
    
        answers = example["answers"]
    
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
    
    def preprocess(example):
        question = example["question"]
        ctxs = example["ctxs"][:10]
        answers = example["answers"]
    
    
        context = "\n".join(ctx["text"] for ctx in ctxs)
        splits_context = sent_tokenize(context)
        context_input_ids = tokenizer(splits_context , add_special_tokens = False , return_attention_mask = False)["input_ids"]
        
    
        question_input_ids = tokenizer(question , return_attention_mask = False)
        question_ids = question_input_ids["input_ids"]
    
    
        all_input_ids = question_ids
        sentence_mask = [ignore_token_id] * (len(question_ids))
        sentence_action_mask = [0] * (len(question_ids))
        sentence_count = 0
    
        for segment_input_ids in context_input_ids:
            all_input_ids.append(sentence_token_id)
            segment_len = len(segment_input_ids)
            all_input_ids.extend(segment_input_ids)
            sentence_action_mask.append(1)
            sentence_action_mask.extend([0] * segment_len)
            sentence_mask.append(sentence_count)
            sentence_mask.extend([sentence_count] * segment_len)
            sentence_count += 1
            
        all_input_ids.append(tokenizer.sep_token_id)
        sentence_action_mask.append(0)
        sentence_mask.append(ignore_token_id)
    
        inputs = {
            "input_ids" : all_input_ids
        }
    
        input_len = max_length - len(all_input_ids)
        sentence_action_mask.extend([0] * input_len)
        sentence_mask.extend([ignore_token_id] * input_len)
    
        inputs = tokenizer.pad(inputs, padding = "max_length", max_length = max_length)
        inputs["sentence_action_mask"] = sentence_action_mask
        inputs["sentence_mask"] = sentence_mask
        inputs["question"] = question
        inputs["answers"] = answers
    
        assert len(inputs["input_ids"]) == len(inputs["sentence_mask"]) == len(inputs["sentence_action_mask"])
    
        return inputs
        
    


dataset = dataset.filter(filter_examples)


dataset = dataset.map(preprocess, remove_columns=dataset.column_names)


if len(dataset) <= hyperparameters["validation_size"]:
    raise ValueError("Validation size must be smaller than the number of examples remaining after filtering.")
train_val_test = dataset.train_test_split(test_size=hyperparameters["validation_size"], seed=hyperparameters["seed"])
train_dataset = train_val_test['train']
validation_dataset = train_val_test['test']

dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
})

print(f"Dataset prepared | Training examples: {len(train_dataset):,} | Validation examples: {len(validation_dataset):,}", flush=True)


def add_index_column(example, idx):
    example["index"] = idx
    return example

# Apply the mapping with with_indices=True
dataset = dataset.map(add_index_column, with_indices=True)


rb = ReplayBuffer(hyperparameters["alpha"])


# if hyperparameters["use_replay"] and hyperparameters["import_empty_dataset"]:
#     rb.import_empty_dataset(dataset["train"])


batch_size = hyperparameters["inference_batch_size"]
def train_collate_fn(batch):
    # Split into question and docs
    indexs = [batch[0]["index"]] * batch_size
    batch_input_ids = [batch[0]["input_ids"]] * batch_size
    batch_attention_mask = [batch[0]["attention_mask"]] * batch_size
    batch_sentence_mask = [batch[0]["sentence_mask"]] * batch_size
    batch_sentence_action_mask = [batch[0]["sentence_action_mask"]] * batch_size
    questions = [batch[0]["question"]] * batch_size
    answers = [batch[0]["answers"]] * batch_size

    return {
        "indexs" : indexs,
        "input_ids" : torch.tensor(batch_input_ids),
        "attention_mask" : torch.tensor(batch_attention_mask),
        "sentence_mask": torch.tensor(batch_sentence_mask),
        "sentence_action_mask": torch.tensor(batch_sentence_action_mask),
        "questions" : questions,
        "answers" : answers,
    }


def eval_collate_fn(batch):
    return {
            "indexs" : [b["index"] for b in batch],
            "input_ids": torch.tensor([b["input_ids"] for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
            "sentence_mask": torch.tensor([b["sentence_mask"] for b in batch]),
            "sentence_action_mask": torch.tensor([b["sentence_action_mask"] for b in batch]),
            "questions": [b["question"] for b in batch],
            "answers": [b["answers"] for b in batch],
        }


train_loader = DataLoader(dataset["train"], batch_size= 1, shuffle=True, collate_fn=train_collate_fn)
eval_loader = DataLoader(dataset["validation"], batch_size=hyperparameters["inference_batch_size"], shuffle=False, collate_fn=eval_collate_fn)


env = Env(hyperparameters["gen_model"] , hyperparameters["generator_device"], hyperparameters["alpha"] , hyperparameters["llm_batch_size"])


def train_deep(agent, optimizer, dataset, backward_batch_size, max_grad_norm):
    agent.train()
    pos_count = 0
    neg_count = 0

    # Map dataset to include labels
    def mapping(sample):
        nonlocal pos_count, neg_count
        action = torch.tensor(sample["action"])
        sentence_action_mask = torch.tensor(sample["sentence_action_mask"])
        labels = torch.where(sentence_action_mask == 1, action, torch.tensor(ignore_token_id))
        neg = torch.sum(labels == 0).item()
        pos = torch.sum(labels == 1).item()
        pos_count += pos
        neg_count += neg
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": labels.tolist()
        }

    # Apply mapping before class weight computation
    dataset = dataset.map(mapping)

    # Compute balanced weights
    all_label = [0] * neg_count + [1] * pos_count
    ce_weight = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=all_label
    )
    ce_weight = torch.tensor(ce_weight, dtype=torch.float32).to(agent.device)

    # Collate function for DataLoader
    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([b["input_ids"] for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch]),
        }

    replay_loader = DataLoader(dataset, batch_size=backward_batch_size, shuffle=True, collate_fn=collate_fn)

    num_training_steps = len(replay_loader)
    progress_bar = tqdm(range(num_training_steps))

    total_loss = 0.0

    for step, batch in enumerate(replay_loader):
        input_ids = batch["input_ids"].to(agent.device)
        attention_mask = batch["attention_mask"].to(agent.device)
        labels = batch["labels"].to(agent.device)

        with torch.amp.autocast(agent.device,dtype=torch.bfloat16):
            outputs = agent(input_ids, attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                weight=ce_weight,
                ignore_index=ignore_token_id,
            )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_description(f"Replay training | Loss: {loss.item():.4f}")

    mean_loss = total_loss / num_training_steps
    return mean_loss


agent = Agent(hyperparameters["filter_model"] , hyperparameters["filter_device"])


optimizer = AdamW(agent.parameters(), lr=hyperparameters["lr"])


def load_latest_model(agent,optimizer,pattern,rb = None):
    model_files = glob.glob(pattern)
    
    if not model_files:
        print("No matching checkpoint found. Starting a new training run.", flush=True)
        return 0
    
    def extract_number(filename):
        basename = os.path.basename(filename)
        digits = ''.join(filter(str.isdigit, basename))
        return int(digits) if digits else -1
    
    model_files.sort(key=extract_number)
    
    latest_model = model_files[-1]
    print(f"Resuming training from checkpoint: {latest_model}", flush=True)
    
    state_dict = torch.load(latest_model, map_location="cpu")
    epoch = state_dict['epoch']
    model_state_dict = state_dict['agent']
    optimizer_state_dict = state_dict['optimizer']
    if (hyperparameters["use_replay"]):
        buffer = state_dict["buffer"]
        rb.import_buffer(buffer)
    
    optimizer.load_state_dict(optimizer_state_dict)
    agent.load_state_dict(model_state_dict)
    epoch = epoch + 1
    return epoch

if hyperparameters["use_replay"]:
    start_epoch = load_latest_model(agent,optimizer,hyperparameters["save_pattern"] , rb)
else:
    start_epoch = load_latest_model(agent,optimizer,hyperparameters["save_pattern"])


if start_epoch == 0:
    agent.eval()
    mean_reward , mean_acc , mean_compress_rate = evaluate(agent, env , eval_loader)
    print_metrics("Initial evaluation", mean_reward, mean_acc, mean_compress_rate)

    if hyperparameters["use_replay"]:
        cold_start_time = time.time()
        mean_reward , mean_acc , mean_compress_rate = replay_cold_start(agent , env , dataset["train"] , rb , hyperparameters["inference_batch_size"])
        total_cold_start_time = time.time() - cold_start_time
        print_metrics("Replay buffer initialization", mean_reward, mean_acc, mean_compress_rate)
        print(f"Replay buffer initialization completed in {total_cold_start_time:.2f} seconds.", flush=True)


class Logger:
    def __init__(self , csv_path):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            with open(csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "eval_acc_score", "eval_reward","eval_compression_rate","train_acc_score","train_reward","train_compress_rate","rl_time","replay_time"])
    
    def save_log(self , epoch , eval_acc_score="" , eval_reward="", eval_compress_rate="", train_acc_score="" , train_reward="" , train_compress_rate="" , rl_time="" , replay_time=""):
        with open(self.csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, eval_acc_score, eval_reward,eval_compress_rate , train_acc_score , train_reward , train_compress_rate,rl_time , replay_time])
        if all(value != "" for value in (eval_reward, eval_acc_score, eval_compress_rate)):
            print_metrics(f"Epoch {epoch + 1} evaluation", eval_reward, eval_acc_score, eval_compress_rate)
        print(f"Evaluation results appended to: {self.csv_path}", flush=True)
        


csv_logger = Logger(hyperparameters["log_path"])


for curr_epoch in range(start_epoch , hyperparameters["epoch"]):
    print(f"Starting epoch {curr_epoch + 1} of {hyperparameters['epoch']}", flush=True)
    rl_start_time = time.time()
    train_mean_reward , train_mean_acc , train_mean_compress_rate = train(agent, env , optimizer, train_loader , rb , hyperparameters["rollout_size"] , hyperparameters["backward_batch_size"] , hyperparameters["n_updates_per_iteration"] , hyperparameters["enf_coefficient"] ,  hyperparameters["clip_value"] , hyperparameters["max_grad_norm"] , hyperparameters["alpha"] , hyperparameters["target_kl"])
    total_rl_time = time.time() - rl_start_time
    print_metrics(f"Epoch {curr_epoch + 1} training", train_mean_reward, train_mean_acc, train_mean_compress_rate)
    mean_reward , mean_acc , mean_compress_rate = evaluate(agent, env , eval_loader)
    csv_logger.save_log(curr_epoch , mean_acc , mean_reward , mean_compress_rate , train_mean_acc , train_mean_reward , train_mean_compress_rate , total_rl_time , 0)
    total_replay_time = 0
    if hyperparameters["use_replay"]:
        dataset = rb.export_dataset()
        if dataset:
            replay_start_time = time.time()
            mean_loss = train_deep(agent, optimizer, dataset, hyperparameters["backward_batch_size"], hyperparameters["max_grad_norm"])
            total_replay_time = time.time() - replay_start_time
            print(f"Replay training completed | Mean loss: {mean_loss:.4f} | Duration: {total_replay_time:.2f} seconds", flush=True)
    mean_reward , mean_acc , mean_compress_rate = evaluate(agent, env , eval_loader)
    csv_logger.save_log(curr_epoch , mean_acc , mean_reward , mean_compress_rate , train_mean_acc , train_mean_reward , train_mean_compress_rate , total_rl_time , total_replay_time)
    if hyperparameters["use_replay"]:
        buffer = rb.export_buffer()
        torch.save({"epoch" : curr_epoch ,"optimizer" : optimizer.state_dict() , "agent" : agent.state_dict() , "buffer" : buffer}, f"agent_best_filter_epoch_{curr_epoch}.pth")
    else:
        torch.save({"epoch" : curr_epoch ,"optimizer" : optimizer.state_dict() , "agent" : agent.state_dict()}, f"agent_best_filter_epoch_{curr_epoch}.pth")


