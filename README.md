# EffComp: Efficient Prompt Compression via Hybrid Reinforcement & Supervised Learning (Findings of AACL 2026)

We present **EffComp**, a framework for compressing retrieved documents for question answering. EffComp uses hybrid reinforcement learning (RL) and supervised learning (SL) to train a sentence selection policy that retains relevant context from retrieved passages. This repository provides the code for training and inference.

🤗 [Model](#model) | 📁 [Data](#data)

## Updates

- **[2026]** EffComp has been accepted to **Findings of AACL 2026**.

## Installation

Our experiments use the following environment:

- Python 3.10.18
- PyTorch 2.8.0

```bash
# Create and activate a new environment
conda create -n effcomp python=3.10.18 -y
conda activate effcomp

# Install PyTorch
python -m pip install torch==2.8.0

# Install requirements
python -m pip install -r requirements.txt

```

## Download

### Model

[Download the EffComp model](#model)

### Data

[Download the datasets](#data)

```json
[
  {
    "question": "Your question",
    "answers": ["Reference answer"],
    "ctxs": [
      {
        "id": "document-id",
        "title": "Document title",
        "text": "Retrieved passage text."
      }
    ]
  }
]
```

Each example requires `question`, `answers`, and `ctxs`, with a `text` field for each passage.

See [examples/input.txt](examples/input.txt) for a complete example. Use valid JSON without trailing commas. For training, prepare a Hugging Face Dataset directory using `save_to_disk()`.

## Inference

Run the following command to rerank retrieved passages and compress them with EffComp. Set `--top-k` to the number of passages for the reranker to select and `--batch-size` to the number of examples processed by the policy in each batch.

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --input data/test.txt \
    --output outputs/compressed.json \
    --model models/compressor \
    --checkpoint models/effcomp.pth \
    --reranker models/reranker \
    --top-k 25 \
    --batch-size 12 \
    --device cuda
```

The script produces compressed contexts for downstream question answering. If no sentence is selected, it returns the retained context.

For all available arguments:

```bash
python inference.py --help
```

## Training

### Pretraining

Pretrain the sentence selection model from a JSON array of examples:

```bash
python pretrain.py \
    --dataset data/train.json \
    --model answerdotai/ModernBERT-base \
    --output-dir outputs/pretrain \
    --epochs 2 \
    --learning-rate 2e-5 \
    --train-batch-size 4 \
    --eval-batch-size 4 \
    --max-length 8192
```

Run `python pretrain.py --help` for all arguments, including preprocessing workers, validation ratio, and token IDs.

### Fine-tuning

Train the sentence selection policy using a dataset with `question`, `answers`, and `ctxs` fields. Set the compression model and answer generator paths as shown below:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --dataset data/train \
    --filter-model models/compressor \
    --gen-model models/generator \
    --epoch 15 \
    --lr 1e-6 \
    --inference-batch-size 8 \
    --backward-batch-size 8 \
    --llm-batch-size 8 \
    --alpha 0.95
```

Best trial memory training is enabled by default. Add `--no-use-replay` to disable it. Use `--validation-size` to adjust the default validation set of 1,000 examples.

Checkpoints are saved as `agent_best_filter_epoch_<index>.pth`. Training automatically resumes from the latest checkpoint matching `--save-pattern`; `--epoch` specifies the total target epoch count.

For all training arguments:

```bash
python finetune.py --help
```

Citation

@misc{tungsununtham2026effcompefficientpromptcompression,
      title={EffComp: Efficient Prompt Compression via Hybrid Reinforcement & Supervised Learning}, 
      author={Romtham Tungsununtham and Piyawat Lertvittayakumjorn and Atiwong Suchato and Peerapon Vateekul},
      year={2026},
      eprint={2611.02766},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2611.02766}, 
}
