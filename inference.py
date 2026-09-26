#!/usr/bin/env python
# coding: utf-8

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress retrieved documents with EffComp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="./dataset/test_hotpotqa_top_30.json", help="Input JSON array (.json or .txt)")
    parser.add_argument("--output", default="./dataset/test_hotpotqa_top_30_bge_top_10_sentence.json", help="Output JSON file")
    parser.add_argument("--model", default="answerdotai/ModernBERT-base", help="Base model and tokenizer path")
    parser.add_argument("--checkpoint", default="agent_best_filter_epoch_5.pth", help="Trained agent checkpoint")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-base", help="BGE reranker path")
    parser.add_argument("--top-k", type=int, default=10, help="Number of reranked passages to retain")
    parser.add_argument("--batch-size", type=int, default=12, help="Agent inference batch size")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--sentence-token-id", type=int, default=50286, help="Sentence marker ID used during training")
    args = parser.parse_args()
    if args.top_k <= 0 or args.batch_size <= 0:
        parser.error("--top-k and --batch-size must be positive")
    return args


args = parse_args()

import os
import json
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification
from FlagEmbedding import FlagReranker

sentence_token_id = args.sentence_token_id
ignore_token_id = -100
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device


class Agent(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.base_model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=2)
        if not trainable:
            self.eval()
            self.requires_grad_(False)

    def forward(self, input_ids, attention_mask=None):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits


agent = Agent(trainable=False).to(DEVICE)
state = torch.load(args.checkpoint, map_location="cpu")
agent.load_state_dict(state["agent"])
tokenizer = AutoTokenizer.from_pretrained(args.model)
reranker = FlagReranker(args.reranker, use_fp16=DEVICE.startswith("cuda"), devices=[DEVICE])
pad_token_id = tokenizer.pad_token_id


def build_single(question, context):
    splits = sent_tokenize(context)
    q_ids = tokenizer(question, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    all_ids = list(q_ids)
    sa_mask = [0] * len(q_ids)
    s_mask = [ignore_token_id] * len(q_ids)
    ctx_ids_list = tokenizer(splits, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    for sid, seg in enumerate(ctx_ids_list):
        all_ids.append(sentence_token_id)
        sa_mask.append(1)
        s_mask.append(sid)
        all_ids.extend(seg)
        sa_mask.extend([0] * len(seg))
        s_mask.extend([sid] * len(seg))
    if tokenizer.sep_token_id is not None:
        all_ids.append(tokenizer.sep_token_id)
        sa_mask.append(0)
        s_mask.append(ignore_token_id)
    return (
        torch.tensor(all_ids, dtype=torch.long),
        torch.tensor(sa_mask, dtype=torch.long),
        torch.tensor(s_mask, dtype=torch.long),
    )


def simple_pad(tensors, pad_value=pad_token_id):
    return pad_sequence(tensors, batch_first=True, padding_value=pad_value)


with open(args.input, "r", encoding="utf-8-sig") as f:
    test_data = json.load(f)

print(f"Compressing retrieved documents: {args.input}", flush=True)
ids_list, sa_list, sm_list = [], [], []
contexts_merged, rerank_times = [], []
global_time = time.time()

# Rerank passages and prepare sentence inputs.
for ex in tqdm(test_data, desc="Reranking and preparing inputs"):
    q = ex["question"]
    ctxs = [c["text"] for c in ex.get("ctxs", [])]
    t0 = time.time()
    pairs = [[q, c] for c in ctxs]
    scores = reranker.compute_score(pairs, normalize=True)
    top_ctxs = [c for c, s in sorted(zip(ctxs, scores), key=lambda x: x[1], reverse=True)[:args.top_k]]
    merged_context = "\n".join(top_ctxs)
    rerank_times.append(time.time() - t0)
    x_ids, x_sa, x_sm = build_single(q, merged_context)
    ids_list.append(x_ids)
    sa_list.append(x_sa)
    sm_list.append(x_sm)
    contexts_merged.append(merged_context)

# Run the sentence selection agent in batches.
summaries, agent_times = [], []
with torch.no_grad():
    for i in tqdm(range(0, len(ids_list), args.batch_size), desc="Compressing document batches"):
        input_ids = simple_pad(ids_list[i:i + args.batch_size]).to(DEVICE)
        sa_mask = simple_pad(sa_list[i:i + args.batch_size], pad_value=0).to(DEVICE)
        s_mask = simple_pad(sm_list[i:i + args.batch_size], pad_value=ignore_token_id).to(DEVICE)
        attn_mask = (input_ids != pad_token_id).long()
        t0 = time.time()
        logits = agent(input_ids=input_ids, attention_mask=attn_mask)
        avg_time = (time.time() - t0) / input_ids.size(0)
        act = logits.argmax(dim=-1)
        for j in range(input_ids.size(0)):
            ids, sa, sm, a = input_ids[j], sa_mask[j], s_mask[j], act[j]
            sel = sm[(sa == 1) & (a == 1)]
            keep = (ids != pad_token_id) & (sa != 1) & torch.isin(sm, sel)
            kept_ids = ids[keep].tolist()
            text = tokenizer.decode(kept_ids, skip_special_tokens=False) if kept_ids else contexts_merged[i + j]
            summaries.append(text)
            agent_times.append(avg_time)

print(f"Compression completed in {time.time() - global_time:.2f} seconds.", flush=True)
for ex, summary, rerank_time, agent_time in zip(test_data, summaries, rerank_times, agent_times):
    ex["summary"] = [summary]
    ex["summary_time"] = rerank_time + agent_time

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
print(f"Results saved to: {args.output}", flush=True)
