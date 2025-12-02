"""
Train a tokenizer in the style of GPT-4 tokenizer.

Notice: the code is copied and adopted from Andrej Karpathy's amazing nanochat repo:
https://github.com/karpathy/nanochat/blob/master/scripts/tok_train.py
"""
import argparse
import pyarrow.parquet as pq
import time

from tokenizer import RustBPETokenizer
from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from pathlib import Path

parser = argparse.ArgumentParser(description="Train a BPE tokenizer with Rust")
parser.add_argument("--dataset_dir", type=Path, default=None, help="Dataset dir (with parquet files)")
parser.add_argument("--vocab_size", type=int, default=65536, help="Vocabulary size (default: 65536 = 2^16)")
parser.add_argument("--output_dir", type=str, default="./nanochat-tokenizer", help="Output directory for trained tokenizer")
args = parser.parse_args()

print(f"vocab_size: {args.vocab_size:,}")

def parquets_iter_batched(dataset_dir):
    """
    Iterate through the dataset directory, in batches of underlying row_groups for efficiency.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    parquet_paths = [d for d in dataset_dir.iterdir() if d.name.endswith(".parquet")]
    for filepath in parquet_paths:
        print(f"Reading from: {filepath}")
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(0, pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

def text_iterator(dataset_dir):
    """
    Flatten the batches into a single iterator.
    """
    for batch in parquets_iter_batched(dataset_dir=dataset_dir):
        for doc in batch:
            yield doc

text_iter = text_iterator(args.dataset_dir)

# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# Save the tokenizer to disk
tokenizer.save(args.output_dir)

# Quick inline sanity check
test_text = """Hallo aus Holzkirchen! Das ist ein Test.
Numbers: 123, 4567, 89
Contractions: Ich, du, es
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

print("Final vocab size:", tokenizer.get_vocab_size())

# Now convert it into a HF Fast Tokenizer
# It really must be separate dir,
# otherwise some stranger things happen...
convert_tiktoken_to_fast(tokenizer.enc, args.output_dir + "-hf")

# Also export config.json and tokenizer_config.json
# Can be probably done with Transformers functions...

tokenizer_config = """{
    "tokenizer_class": "PreTrainedTokenizerFast",
    "bos_token": "<|bos|>",
    "eos_token": "<|assistant_end|>",
    "pad_token": "<|assistant_end|>",
    "additional_special_tokens": [
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|python_start|>",
        "<|python_end|>",
        "<|output_start|>",
        "<|output_end|>"
    ],
    "model_input_names": [
        "input_ids",
        "attention_mask"
    ]
}
"""

# Based on LLäMmlein:
# https://huggingface.co/LSX-UniWue/LLaMmlein_120M
config_json ="""{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 65527,
  "eos_token_id": 65529,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 2048,
  "max_position_embeddings": 2048,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_key_value_heads": 4,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.57.1",
  "use_cache": true,
  "vocab_size": 65536
}"""

with open(args.output_dir + "-hf" + "/" + "config.json", "wt") as f_out:
    f_out.write(config_json)

with open(args.output_dir + "-hf" + "/" + "tokenizer_config.json", "wt") as f_out:
    f_out.write(tokenizer_config)
