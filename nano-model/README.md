# Nano Model

This section shows how to pretrain a Nano Model (LLama-based, 125M parameters).

Three tokenizers will be trained to perform various ablation experiments.

## Setup

In order to build the nanochat tokenizer - taken from [the nanochat repo](https://github.com/karpathy/nanochat) - we need to install various dependencies, that are documented in this section.

First, change to the correct folder:

```bash
cd nano-model
```

then the following commands will install all necessary packages (both Rust and Python):

```bash
# Install cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv

# Install all dependencies
uv sync --extra cpu

# Activate venv
source .venv/bin/activate

# Build rust tokenizer package
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## Vocab Corpus

We download the first 8 parquet files from the German nanochat corpus for our tokenizer pretraining:

```bash
mkdir vocab-corpus

hf download --local-dir ./vocab-corpus stefan-it/nanochat-german-data shard_0000{0..7}.parquet

```

## Tokenizer Training

In this section, we train three different tokenizers in order to use them for further ablations.

### nanochat Tokenizer

The nanochat tokenizer training can be started with:

```bash
python3 nanochat-tokenizer/train_tokenizer.py \
    --dataset_dir ./vocab-corpus \
    --output_dir nanochat-tokenizer
```

It outputs:

```
vocab_size: 65,536
Reading from: vocab-corpus/shard_00007.parquet
Reading from: vocab-corpus/shard_00005.parquet
Reading from: vocab-corpus/shard_00001.parquet
Reading from: vocab-corpus/shard_00003.parquet
Reading from: vocab-corpus/shard_00004.parquet
Reading from: vocab-corpus/shard_00006.parquet
Reading from: vocab-corpus/shard_00000.parquet
Reading from: vocab-corpus/shard_00002.parquet
Training time: 144.90s
Saved tokenizer encoding to nanochat-tokenizer/tokenizer.pkl
Final vocab size: 65536
```

Additionally, also a HF fast tokenizer is written to `nanochat-tokenizer-hf`, which can be test with:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./nanochat-tokenizer-hf")

tokenizer.tokenize("Willkommen in Holzkirchen, südlich von München!")

# Outputs:
# ['Willkommen', 'Ġin', 'ĠHolz', 'kirchen', ',', 'ĠsÃ¼dlich', 'Ġvon', 'ĠMÃ¼nchen', '!']
```

### GPT-NeoX Tokenizer

The GPT-NeoX tokenizer code was adopted from [this repository](https://github.com/EleutherAI/gpt-neox).
