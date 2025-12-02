# Nano Model

This section shows how to pretrain a Nano Model (LLama-based, 125M parameters).

Two tokenizers (nanochat and GPT-NeoX) will be trained to perform the first ablation experiments.

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

huggingface-cli download \
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

Additionally, a HF fast tokenizer is written to `nanochat-tokenizer-hf`, which can be tested with:

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

So the training is slightly slower than the nanochat tokenizer.

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

```bash
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

## Pretraining Corpus

We use the first 240 parquet files from the German nanochat corpus for constructing our pretraining corpus.

### TensorFlow Datasets (TFDS)

In general, MaxText has support for [Grain](https://maxtext.readthedocs.io/en/latest/guides/data_input_pipeline/data_input_grain.html), [Hugging Face Model Hub](https://maxtext.readthedocs.io/en/latest/guides/data_input_pipeline/data_input_hf.html) and [TensorFlow Datasets](https://maxtext.readthedocs.io/en/latest/guides/data_input_pipeline/data_input_tfds.html).

For our project we use TFDS for the following reasons:

* HF Hub has too many limitations (sometimes 504 errors, only 1 epoch is currently supported)
* Grain would require a mounted GCP bucket (I have no experience with how stable that is)

So in the next section, we show how to convert a Hugging Face Dataset to a TensorFlow dataset (TFDS). That dataset will be uploaded to a GCP bucket so it can be used for MaxText training.

#### TFDS Init

First, we create a new dataset with the [TFDS CLI](https://www.tensorflow.org/datasets/add_dataset):

```bash
tfds new german_maxtext_nano_data
```

**Note**: We will create a small dataset for debugging purposes first.

#### TFDS Implementation

The previous `tfds new` command created a new folder structure for our dataset. Now, we need to implement the dataset builder logic:

* The original dataset is fetched from the Hugging Face Model Hub using the `load_dataset()` method
* The `text` column of each dataset entry is read and returned in a TFDS compatible format

For this, we need to modify the `german_maxtext_nano_data/german_maxtext_nano_data_dataset_builder.py` file:

```python
"""german_maxtext_nano_data dataset."""

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for german_maxtext_nano_data dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'text': tfds.features.Text(),
        }),
        supervised_keys=None,
        homepage='https://huggingface.co/datasets/german-maxtext-slms/nano-training-data',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Load dataset from Hugging Face
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset("german-maxtext-slms/nano-training-data")

    # Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(ds['train']),
    }

  def _generate_examples(self, dataset):
    """Yields examples."""
    for idx, row in enumerate(dataset):
      yield idx, {
          'text': row['text'],
      }
```

It is a very easy to understand and compact builder.

#### TFDS Build

After writing the very compact builder script, it is time to build the actual TFDS dataset using:

```bash
cd german_maxtext_nano_data
tfds build
```

The final created TFDS is located under `$HOME/tensorflow_datasets/german_maxtext_nano_data`. This folder needs to be uploaded to a GCP bucket:

```bash
gsutil -m cp -r $HOME/tensorflow_datasets/german_maxtext_nano_data gs://german-maxtext
```

**Notice:** Please adjust the GCP bucket name.

## TPU Setup

This section shows how to set up a TPU VM to start LLM pretraining. It is mainly inspired by the [official documentation](https://docs.cloud.google.com/tpu/docs/v6e-training?hl=en).

### Environment Variables

First, we need to set the GCP project id and TPU zone:

```bash
export PROJECT_ID=XXX
export ZONE=us-central2-b
export TPU_NAME=german-maxtext
export ACCELERATOR_TYPE=v4-32
export RUNTIME_VERSION=v2-alpha-tpuv4-pod
```

Additionally, we also experiment with v6e-8:

```bash
export PROJECT_ID=XXX
export ZONE=europe-west4-a
export TPU_NAME=german-maxtext
export ACCELERATOR_TYPE=v6e-8
export RUNTIME_VERSION=v2-alpha-tpuv6e
```

### TPU VM Creation

Then the TPU VM can be created via the [queued resource manager](https://cloud.google.com/tpu/docs/queued-resources):

```bash
# https://docs.cloud.google.com/tpu/docs/spot?hl=en
gcloud compute tpus queued-resources create german-maxtext-resource \
  --node-id $TPU_NAME \
  --project $PROJECT_ID \
  --zone $ZONE \
  --accelerator-type $ACCELERATOR_TYPE \
  --runtime-version $RUNTIME_VERSION \
  --spot
```

You can check the creation status with:

```bash
gcloud alpha compute tpus queued-resources list --project $PROJECT_ID --zone $ZONE
```

The different states are `WAITING_FOR_RESOURCES`, `PROVISIONING` and `ACTIVE`.

### Dependencies

As soon as the TPU VM has reached the `ACTIVE` state, we can SSH into the VM and start a `tmux` session:

```bash
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
   --project=${PROJECT_ID} \
   --zone ${ZONE} \
   --worker=0

tmux
```

Then all necessary dependencies can be installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Check that uv is properly installed
uv --version

# We use a forked repo that comes with all necessary configs
git clone https://github.com/stefan-it/maxtext.git
cd maxtext
git checkout german-maxtext-slms

uv venv --python 3.12 --seed maxtext_venv
source maxtext_venv/bin/activate

uv pip install -e .[tpu] --resolution=lowest
```

### Demo Training

It is highly recommended to test the TPU VM setup including MaxText. This can be done by starting a demo run:

```bash
export RUN_NAME=demo-run
export GCP_BUCKET=gs://german-maxtext/demo-run-v4-32
python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=$RUN_NAME \
  base_output_directory=$GCP_BUCKET \
  dataset_type=synthetic \
  steps=100
```

## Ablations

### nano Model (nanochat Tokenizer)

The first ablation model uses our previously trained nanochat tokenizer.

As hyper-parameters we use the same as proposed in the [LLäMmlein](https://arxiv.org/abs/2411.11171) paper:

| Parameter      | MaxText Parameter Name                                                    | Value                |
|:---------------|:--------------------------------------------------------------------------|---------------------:|
| Steps          | `steps`                                                                   | 14,500               |
| Learning Rate  | `learning_rate`                                                           | 6e-06                |
| Batch Size     | `per_device_batch_size` x `gradient_accumulation_steps` x Number of Chips | 32 x 4 x 8 = 1024    |
| Context Length | `max_target_length`                                                       | 2048                 |

Notice: We trained the model on a v6e-8 TPU.

```bash
export RUN_NAME=nano-nanochat-tokenizer-ablation-2
export DATASET_PATH=gs://german-maxtext

python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=$RUN_NAME \
  base_output_directory=gs://german-maxtext/$RUN_NAME \
  dataset_type=tfds \
  dataset_path=${DATASET_PATH} \
  dataset_name=german_maxtext_nano_data \
  train_split=train \
  async_checkpointing=false \
  model_name='nano-german-slm' \
  learning_rate=6e-06 \
  per_device_batch_size=32 \
  gradient_accumulation_steps=4 \
  steps=14500 \
  max_target_length=2048 \
  packing=false \
  checkpoint_period=10000 \
  tokenizer_type=huggingface tokenizer_path=german-maxtext-slms/nano-nanochat-tokenizer
```

### nano Model (GPT-NeoX Tokenizer)

The second ablation model uses our previously trained GPT-NeoX tokenizer.

As hyper-parameters we use the same as proposed in the [LLäMmlein](https://arxiv.org/abs/2411.11171) paper:

| Parameter      | MaxText Parameter Name                                                    | Value                 |
|:---------------|:--------------------------------------------------------------------------|----------------------:|
| Steps          | `steps`                                                                   | 14,500                |
| Learning Rate  | `learning_rate`                                                           | 6e-06                 |
| Batch Size     | `per_device_batch_size` x `gradient_accumulation_steps` x Number of Chips | 32 x 2 x 4 x 4 = 1024 |
| Context Length | `max_target_length`                                                       | 2048                  |

Notice: We trained the model on a v4-32 TPU Pod.

```bash
export RUN_NAME=nano-gpt-neox-tokenizer-ablation-pod-1
export DATASET_PATH=gs://german-maxtext-2

python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=$RUN_NAME \
  base_output_directory=${DATASET_PATH}/$RUN_NAME \
  dataset_type=tfds \
  dataset_path=${DATASET_PATH} \
  dataset_name=german_maxtext_nano_data \
  train_split=train \
  async_checkpointing=false \
  model_name='nano-german-slm' \
  learning_rate=6e-06 \
  per_device_batch_size=32 \
  gradient_accumulation_steps=2 \
  steps=466509 \
  max_target_length=2048 \
  packing=false \
  checkpoint_period=10000 \
  tokenizer_type=huggingface tokenizer_path=german-maxtext-slms/nano-neox-tokenizer
```

## Ablations - Different Learning Rates

In this section, we try out different learning rates for different tokenizers.

### nano Model (nanochat Tokenizer)

As hyper-parameters we use the same as proposed in the [LLäMmlein](https://arxiv.org/abs/2411.11171) paper:

| Parameter      | MaxText Parameter Name                                                    | Value                 |
|:---------------|:--------------------------------------------------------------------------|----------------------:|
| Steps          | `steps`                                                                   | 14,500                |
| Learning Rate  | `learning_rate`                                                           | 4e-06                 |
| Batch Size     | `per_device_batch_size` x `gradient_accumulation_steps` x Number of Chips | 32 x 2 x 4 x 4 = 1024 |
| Context Length | `max_target_length`                                                       | 2048                  |

Notice: We trained the model on a v4-32 TPU Pod.

```bash
export RUN_NAME=nano-nanochat-tokenizer-ablation-lr4e-06-1
export DATASET_PATH=gs://german-maxtext-2

python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=$RUN_NAME \
  base_output_directory=gs://german-maxtext/$RUN_NAME \
  dataset_type=tfds \
  dataset_path=${DATASET_PATH} \
  dataset_name=german_maxtext_nano_data \
  train_split=train \
  async_checkpointing=false \
  model_name='nano-german-slm' \
  learning_rate=4e-06 \
  per_device_batch_size=32 \
  gradient_accumulation_steps=2 \
  steps=14500 \
  max_target_length=2048 \
  packing=false \
  checkpoint_period=10000 \
  tokenizer_type=huggingface tokenizer_path=german-maxtext-slms/nano-nanochat-tokenizer
```

### nano Model (GPT-NeoX Tokenizer)

As hyper-parameters we use the same as proposed in the [LLäMmlein](https://arxiv.org/abs/2411.11171) paper:

| Parameter      | MaxText Parameter Name                                                    | Value                 |
|:---------------|:--------------------------------------------------------------------------|----------------------:|
| Steps          | `steps`                                                                   | 14,500                |
| Learning Rate  | `learning_rate`                                                           | 4e-06                 |
| Batch Size     | `per_device_batch_size` x `gradient_accumulation_steps` x Number of Chips | 32 x 2 x 4 x 4 = 1024 |
| Context Length | `max_target_length`                                                       | 2048                  |

Notice: We trained the model on a v4-32 TPU Pod.

```bash
export RUN_NAME=nano-gpt-neox-tokenizer-ablation-pod-1-lr4e-06-1
export DATASET_PATH=gs://german-maxtext-2

python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=$RUN_NAME \
  base_output_directory=gs://german-maxtext/$RUN_NAME \
  dataset_type=tfds \
  dataset_path=${DATASET_PATH} \
  dataset_name=german_maxtext_nano_data \
  train_split=train \
  async_checkpointing=false \
  model_name='nano-german-slm' \
  learning_rate=4e-06 \
  per_device_batch_size=32 \
  gradient_accumulation_steps=2 \
  steps=14500 \
  max_target_length=2048 \
  packing=false \
  checkpoint_period=10000 \
  tokenizer_type=huggingface tokenizer_path=german-maxtext-slms/nano-neox-tokenizer
```

## Checkpoint Conversion

### nano Model (nanochat Tokenizer)

```bash
JAX_PLATFORMS=cpu python3 src/MaxText/utils/ckpt_scripts/llama_mistral_mixtral_orbax_to_hf.py \
  src/MaxText/configs/base.yml \
  base_output_directory=gs://german-maxtext/nano-nanochat-tokenizer-ablation-2 \
  load_parameters_path=gs://german-maxtext/nano-nanochat-tokenizer-ablation-2/nano-nanochat-tokenizer-ablation-2/checkpoints/14499/items \
  run_name=convert_to_hf model_name=nano-german-slm hf_model_path=./nano-nanochat-ablation-hf
```

### nano Model (nanochat Tokenizer, lr=4e-06)

```bash
JAX_PLATFORMS=cpu python3 src/MaxText/utils/ckpt_scripts/llama_mistral_mixtral_orbax_to_hf.py \
  src/MaxText/configs/base.yml \
  base_output_directory=gs://german-maxtext/nano-nanochat-tokenizer-ablation-lr4e-06-1 \
  load_parameters_path=gs://german-maxtext/nano-nanochat-tokenizer-ablation-lr4e-06-1/nano-nanochat-tokenizer-ablation-lr4e-06-1/checkpoints/14499/items \
  run_name=convert_to_hf model_name=nano-german-slm hf_model_path=./nano-nanochat-ablation-lr4e-06-hf
```

### nano Model (GPT-NeoX Tokenizer)

```bash
JAX_PLATFORMS=cpu python3 src/MaxText/utils/ckpt_scripts/llama_mistral_mixtral_orbax_to_hf.py \
  src/MaxText/configs/base.yml \
  base_output_directory=gs://german-maxtext-2/nano-gpt-neox-tokenizer-ablation-pod-2 \
  load_parameters_path=gs://german-maxtext-2/nano-gpt-neox-tokenizer-ablation-pod-2/nano-gpt-neox-tokenizer-ablation-pod-2/checkpoints/14499/items \
  run_name=convert_to_hf model_name=nano-german-slm hf_model_path=./nano-neox-ablation-hf
```

### nano Model (GPT-NeoX Tokenizer, lr=4e-06)

```bash
JAX_PLATFORMS=cpu python3 src/MaxText/utils/ckpt_scripts/llama_mistral_mixtral_orbax_to_hf.py \
  src/MaxText/configs/base.yml \
  base_output_directory=gs://german-maxtext/nano-gpt-neox-tokenizer-ablation-pod-1-lr4e-06-1 \
  load_parameters_path=gs://german-maxtext/nano-gpt-neox-tokenizer-ablation-pod-1-lr4e-06-1/nano-gpt-neox-tokenizer-ablation-pod-1-lr4e-06-1/checkpoints/14499/items \
  run_name=convert_to_hf model_name=nano-german-slm hf_model_path=./nano-neox-ablation-lr4e-06-hf
```

## LM Eval

### Llämmlein 120M

```bash
lm_eval --model hf --model_args pretrained="LSX-UniWue/LLaMmlein_120M" \
    --tasks "arc_de,hellaswag_de,m_mmlu_de,truthfulqa_de_mc1,truthfulqa_de_mc2" \
    --device cuda:0 \
    --batch_size auto \
    --trust_remote_code \
    --log_samples \
    --output_path ./results-LLaMmlein_120M
```

|      Tasks      |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_de           |      2|none  |     0|acc     |↑  |0.1942|±  |0.0116|
|                 |       |none  |     0|acc_norm|↑  |0.2301|±  |0.0123|
|hellaswag_de     |      1|none  |     0|acc     |↑  |0.2945|±  |0.0047|
|                 |       |none  |     0|acc_norm|↑  |0.3178|±  |0.0048|
|m_mmlu_de        |      0|none  |     0|acc     |↑  |0.2285|±  |0.0036|
|truthfulqa_de_mc1|      1|none  |     0|acc     |↑  |0.2310|±  |0.0150|
|truthfulqa_de_mc2|      1|none  |     0|acc     |↑  |0.4055|±  |0.0153|

### nano (nanochat tokenizer, ablation I)

```bash
lm_eval --model hf --model_args pretrained="german-maxtext-slms/nano-nanochat-ablation" \
    --tasks "arc_de,hellaswag_de,m_mmlu_de,truthfulqa_de_mc1,truthfulqa_de_mc2" \
    --device cuda:0 \
    --batch_size auto \
    --trust_remote_code \
    --log_samples \
    --output_path ./results-nano-nanochat-ablation
```

|      Tasks      |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_de           |      2|none  |     0|acc     |↑  |0.1959|±  |0.0116|
|                 |       |none  |     0|acc_norm|↑  |0.2532|±  |0.0127|
|hellaswag_de     |      1|none  |     0|acc     |↑  |0.2523|±  |0.0045|
|                 |       |none  |     0|acc_norm|↑  |0.2500|±  |0.0045|
|m_mmlu_de        |      0|none  |     0|acc     |↑  |0.2347|±  |0.0037|
|truthfulqa_de_mc1|      1|none  |     0|acc     |↑  |0.2525|±  |0.0155|
|truthfulqa_de_mc2|      1|none  |     0|acc     |↑  |0.5157|±  |0.0164|

### nano (nanochat tokenizer, lr=4e-06, ablation II)

```bash
lm_eval --model hf --model_args pretrained="german-maxtext-slms/nano-nanochat-lr4e-06-ablation" \
    --tasks "arc_de,hellaswag_de,m_mmlu_de,truthfulqa_de_mc1,truthfulqa_de_mc2" \
    --device cuda:0 \
    --batch_size auto \
    --trust_remote_code \
    --log_samples \
    --output_path ./results-nano-nanochat-lr4e-06-ablation
```

|      Tasks      |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_de           |      2|none  |     0|acc     |↑  |0.1925|±  |0.0115|
|                 |       |none  |     0|acc_norm|↑  |0.2558|±  |0.0128|
|hellaswag_de     |      1|none  |     0|acc     |↑  |0.2498|±  |0.0045|
|                 |       |none  |     0|acc_norm|↑  |0.2488|±  |0.0045|
|m_mmlu_de        |      0|none  |     0|acc     |↑  |0.2325|±  |0.0037|
|truthfulqa_de_mc1|      1|none  |     0|acc     |↑  |0.2538|±  |0.0155|
|truthfulqa_de_mc2|      1|none  |     0|acc     |↑  |0.5217|±  |0.0164|

### nano (GPT-NeoX tokenizer, ablation I)

```bash
lm_eval --model hf --model_args pretrained="german-maxtext-slms/nano-neox-ablation" \
    --tasks "arc_de,hellaswag_de,m_mmlu_de,truthfulqa_de_mc1,truthfulqa_de_mc2" \
    --device cuda:0 \
    --batch_size auto \
    --trust_remote_code \
    --log_samples \
    --output_path ./results-nano-neox-ablation
```

|      Tasks      |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_de           |      2|none  |     0|acc     |↑  |0.2019|±  |0.0117|
|                 |       |none  |     0|acc_norm|↑  |0.2601|±  |0.0128|
|hellaswag_de     |      1|none  |     0|acc     |↑  |0.2511|±  |0.0045|
|                 |       |none  |     0|acc_norm|↑  |0.2432|±  |0.0044|
|m_mmlu_de        |      0|none  |     0|acc     |↑  |0.2365|±  |0.0037|
|truthfulqa_de_mc1|      1|none  |     0|acc     |↑  |0.2589|±  |0.0156|
|truthfulqa_de_mc2|      1|none  |     0|acc     |↑  |0.5179|±  |0.0165|

### nano (GPT-NeoX tokenizer, lr=4e-06, ablation II)

```bash
lm_eval --model hf --model_args pretrained="german-maxtext-slms/nano-neox-lr4e-06-ablation" \
    --tasks "arc_de,hellaswag_de,m_mmlu_de,truthfulqa_de_mc1,truthfulqa_de_mc2" \
    --device cuda:0 \
    --batch_size auto \
    --trust_remote_code \
    --log_samples \
    --output_path ./results-nano-neox-lr4e-06-ablation
```

|      Tasks      |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_de           |      2|none  |     0|acc     |↑  |0.2044|±  |0.0118|
|                 |       |none  |     0|acc_norm|↑  |0.2609|±  |0.0128|
|hellaswag_de     |      1|none  |     0|acc     |↑  |0.2515|±  |0.0045|
|                 |       |none  |     0|acc_norm|↑  |0.2464|±  |0.0045|
|m_mmlu_de        |      0|none  |     0|acc     |↑  |0.2414|±  |0.0037|
|truthfulqa_de_mc1|      1|none  |     0|acc     |↑  |0.2640|±  |0.0157|
|truthfulqa_de_mc2|      1|none  |     0|acc     |↑  |0.5210|±  |0.0165|

### Overview

| Model                        | arc_de (acc) | arc_de (acc_norm) | hellaswag_de (acc) | hellaswag_de (acc_norm) | m_mmlu_de (acc) | truthfulqa_de_mc1 (acc) | truthfulqa_de_mc2 (acc) | Avg.   |
|:-----------------------------|-------------:|------------------:|-------------------:|------------------------:|----------------:|------------------------:|------------------------:|-------:|
| Llämmlein 120M               | 0.1942       | 0.2301            | 0.2945             | 0.3178                  | 0.2285          | 0.2310                  | 0.4055                  | 0.2717 |
| nano + nanochat tokenizer I  | 0.1959       | 0.2532            | 0.2523             | 0.2500                  | 0.2347          | 0.2525                  | 0.5157                  | 0.2792 |
| nano + nanochat tokenizer II | 0.1925       | 0.2558            | 0.2498             | 0.2488                  | 0.2325          | 0.2538                  | 0.5217                  | 0.2793 |
| nano + GPT-NeoX tokenizer I  | 0.2019       | 0.2601            | 0.2511             | 0.2432                  | 0.2365          | 0.2589                  | 0.5179                  | 0.2814 |
| nano + GPT-NeoX tokenizer II | 0.2044       | 0.2609            | 0.2515             | 0.2464                  | 0.2414          | 0.2640                  | 0.5210                  | 0.2842 |

Overall, it seems that the GPT-NeoX tokenizer achieves better results. Thus, we will select this tokenizer as the base for future experiments.
