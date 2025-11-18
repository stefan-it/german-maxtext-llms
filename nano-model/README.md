# Nano Model

This section shows how to pretrain a Nano Model (LLama-based, 125M parameters).

Two tokenizers (nanochat and GPT-NeoX) will be trained to perform various ablation experiments.

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

hf download \
    --repo-type dataset \
    --local-dir ./vocab-corpus stefan-it/nanochat-german-data shard_0000{0..7}.parquet
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
Reading from: vocab-corpus/shard_00005.parquet
Reading from: vocab-corpus/shard_00003.parquet
Reading from: vocab-corpus/shard_00000.parquet
Reading from: vocab-corpus/shard_00004.parquet
Reading from: vocab-corpus/shard_00001.parquet
Reading from: vocab-corpus/shard_00007.parquet
Reading from: vocab-corpus/shard_00006.parquet
Reading from: vocab-corpus/shard_00002.parquet
Training time: 60.87s
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

The trained tokenizer can also be found on the [Model Hub](https://huggingface.co/german-maxtext-slms/nano-nanochat-tokenizer).

### GPT-NeoX Tokenizer

The GPT-NeoX tokenizer code was adopted from [this repository](https://github.com/EleutherAI/gpt-neox).

The training can be started with:

```bash
python3 gpt-neox-tokenizer/train_tokenizer.py \
    --dataset_dir ./vocab-corpus/
```

At the end it outputs:

```
Training time: 61.88s
```

So the training is a bit slower than the nanochat tokenizer.

The trained GPT-NeoX tokenizer can also be found on the [Model Hub](https://huggingface.co/german-maxtext-slms/nano-neox-tokenizer).

## Tokenizer Evaluation

We use the [GerTokEval](https://github.com/stefan-it/german-tokenizer-benchmark/tree/master) benchmark and extended the default configuration under `./german-configs/tokenizer/llms/tokenizer_config.json` to:

```json
{
  "german_gpt2": {
    "class": "huggingface",
    "path": "dbmdz/german-gpt2"
  },
  "llämmlein": {
    "class": "huggingface",
    "path": "LSX-UniWue/LLaMmlein_120M"
  },
  "teuken": {
    "class": "huggingface",
    "path": "openGPT-X/Teuken-7B-instruct-v0.6"
  },
  "nanochat_german": {
    "class": "huggingface",
    "path": "stefan-it/nanochat-german-v1"
  },
  "nano_nanochat": {
    "class": "huggingface",
    "path": "german-maxtext-slms/nano-nanochat-tokenizer"
  },
  "nano_neox": {
    "class": "huggingface",
    "path": "german-maxtext-slms/nano-neox-tokenizer"
  }
}
```

Then we started the evaluation framework with:

```
python3 scripts/run_tokenizer_analysis.py \
  --samples-per-lang 153035 \
  --tokenizer-config german-configs/tokenizer/llms/tokenizer_config.json \
  --language-config german-configs/datasets/dataset-configs.json \
  --per-language-plots \
  --output-dir german-llms-results \
  --generate-latex-tables
```

This shows the following results:

| Tokenizer       | Fertility | Vocab Utilization | Type-Token Ratio | Token Length | Tokens per Line |
|-----------------|-----------|-------------------|------------------|--------------|-----------------|
| german_gpt2     | 0.212     | 93.3%             | 0.0067           | 4.94         | 28.1            |
| llämmlein       | 0.235     | **97.1%**         | 0.0042           | 4.53         | 29.8            |
| teuken          | 0.252     | 19.0%             | 0.0056           | 4.17         | **34.1**        |
| nanochat_german | 0.207     | 91.5%             | 0.0086           | 5.09         | 27.7            |
| nano_nanochat   | 0.207     | 91.6%             | 0.0086           | 5.09         | 27.7            |
| nano_neox       | **0.204** | 92.4%             | **0.0090**       | **5.15**     | 26.9            |

It seems that our GPT-NeoX tokenizer does a great job!
