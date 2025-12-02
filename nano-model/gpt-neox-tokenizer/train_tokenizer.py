"""
Train a tokenizer in the style of GPT-NeoX tokenizer.

Notice: the code is and adopted from Eleuther AI's GPT-NeoX repo:
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/tokenizer
"""
import argparse
import json
import pyarrow.parquet as pq
import time

from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFKC


parser = argparse.ArgumentParser(description="Train a GPT-NeoX tokenizer with Rust")
parser.add_argument("--dataset_dir", type=Path, default=None, help="Dataset dir (with parquet files)")
parser.add_argument("--vocab_size", type=int, default=65536, help="Vocabulary size (default: 65536 = 2^16)")
parser.add_argument("--output_dir", type=Path, default="./neox-tokenizer", help="Output directory for trained tokenizer")
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

model = models.BPE()

tokenizer = Tokenizer(model)

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
tokenizer.normalizer = NFKC()

# And then train
trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    # We use same special tokens as for our nanochat tokenizer
    special_tokens=[
        # every document begins with the Beginning of Sequence (BOS) token that delimits documents
        "<|bos|>",
        # tokens below are only used during finetuning to render Conversations into token ids
        "<|user_start|>", # user messages
        "<|user_end|>",
        "<|assistant_start|>", # assistant messages
        "<|assistant_end|>",
        "<|python_start|>", # assistant invokes python REPL tool
        "<|python_end|>",
        "<|output_start|>", # python REPL outputs back to assistant
        "<|output_end|>",
    ]
)

# Train the tokenizer
t0 = time.time()
tokenizer.train_from_iterator(text_iter, trainer)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# Save the tokenizer to disk
args.output_dir.mkdir(parents=True, exist_ok=True)

tokenizer.save(str(args.output_dir / "tokenizer.json"), pretty=True)

# Also export config.json and tokenizer_config.json
# Can be probably done with Transformers functions...

tokenizer_config = {
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

# Based on LLäMmlein:
# https://huggingface.co/LSX-UniWue/LLaMmlein_120M
config = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "max_position_embeddings": 2048,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "num_key_value_heads": 4,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.57.1",
    "use_cache": True,
    "vocab_size": args.vocab_size
}

with open(str(args.output_dir / "config.json"), "wt") as f_out:
    json.dump(config, f_out, indent=2)

with open(str(args.output_dir / "tokenizer_config.json"), "wt") as f_out:
    json.dump(tokenizer_config, f_out, indent=4)
