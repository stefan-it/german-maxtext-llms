# Brötchen SLM

## Setup

In order to start our DataTrove pipelines, we need to install various dependencies, that are documented in this section.

First, change to the correct folder:

```bash
cd brötchen-model
```

then the following commands will install all necessary packages (both Rust and Python):

```bash
# Install uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

source ~/.bashrc

uv sync

# Activate venv
source .venv/bin/activate
```

## Datasets

For our dataset preparation pipeline we use the great [DataTrove](https://github.com/huggingface/datatrove) library.

### Vocab Dataset

First, we download the necessary parquet files (`brötchen-model` as root folder):

```bash
# FineWeb2 (German)
hf download HuggingFaceFW/fineweb-2 \
  --repo-type dataset \
  --revision af9c13333eb981300149d5ca60a8e9d659b276b9 \
  --include "data/deu_Latn/train/000_0000[0-9].parquet" \
  --max-workers 1 \
  --local-dir ./fineweb2_german

# FinePdfs
hf download HuggingFaceFW/finepdfs \
  --repo-type dataset \
  --revision d8e85441529983be986a6eb9e0316627c8035e6d \
  --include "data/deu_Latn/train/000_0000[0-9].parquet" \
  --max-workers 1 \
  --local-dir ./finepdfs_german

# FineWeb (English)
hf download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --revision 9bb295ddab0e05d785b879661af7260fed5140fc \
  --include "sample/10BT/*.parquet" \
  --max-workers 1 \
  --local-dir ./fineweb_english

# FineWeb-Edu (English)
hf download HuggingFaceFW/fineweb-edu \
  --repo-type dataset \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --include "sample/10BT/*.parquet" \
  --max-workers 1 \
  --local-dir ./fineweb_edu_english

# FinePdfs
hf download HuggingFaceFW/finepdfs \
  --repo-type dataset \
  --revision d8e85441529983be986a6eb9e0316627c8035e6d \
  --include "data/eng_Latn/train/000_0000[0-9].parquet" \
  --max-workers 1 \
  --local-dir ./finepdfs_english
```

### Training Dataset

```bash
# FineWeb2 (German)
hf download HuggingFaceFW/fineweb-2 \
  --repo-type dataset \
  --revision af9c13333eb981300149d5ca60a8e9d659b276b9 \
  --include "data/deu_Latn/train/000_0000[0-9].parquet" \
  --include "data/deu_Latn/train/000_0001[0-9].parquet" \
  --include "data/deu_Latn/train/000_0002[0-9].parquet" \
  --local-dir ./fineweb2_german

# FinePdfs
hf download HuggingFaceFW/finepdfs \
  --repo-type dataset \
  --revision d8e85441529983be986a6eb9e0316627c8035e6d \
  --include "data/deu_Latn/train/000_0000[0-9].parquet" \
  --include "data/deu_Latn/train/000_0001[0-9].parquet" \
  --include "data/deu_Latn/train/000_0002[0-9].parquet" \
  --local-dir ./finepdfs_german

# FineWeb (English)
hf download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --revision 9bb295ddab0e05d785b879661af7260fed5140fc \
  --include "sample/10BT/*.parquet" \
  --local-dir ./fineweb_english

# FineWeb-Edu (English)
hf download HuggingFaceFW/fineweb-edu \
  --repo-type dataset \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --include "sample/10BT/*.parquet" \
  --local-dir ./fineweb_edu_english

# FinePdfs
hf download HuggingFaceFW/finepdfs \
  --repo-type dataset \
  --revision d8e85441529983be986a6eb9e0316627c8035e6d \
  --include "data/eng_Latn/train/000_0000[0-9].parquet" \
  --local-dir ./finepdfs_english
```

## Vocab Data

We want to use 2B subtokens (measured with our GPT-NeoX tokenizer from the nano model) as our vocab corpus from the following datasets:

| Dataset               | Percentage | Tokens   | Language |
|-----------------------|------------|---------:|----------|
| FineWeb2 (German)     | 30%        | 600M     | German   |
| FinePdfs (German)     | 50%        |   1B     | German   |
| FineWeb (English)     | 5%         | 100M     | English  |
| FineWeb-Edu (English) | 5%         | 100M     | English  |
| FinePdfs (English)    | 10%        | 200M     | English  |

The pipeline for creating our vocab data can be started with:

```bash
python3 -m pipeline.start_pipeline --config ./pipeline/configs/vocab-corpus.yaml
```

The overall subtoken stats can be retrieved via:

```bash
python3 -m pipeline.count_subtokens --config ./pipeline/configs/vocab-corpus.yaml
```

| Dataset name        | Total Subtokens   |
|---------------------|------------------:|
| finepdfs_english    | 200,003,169       |
| finepdfs_german     | 1,000,020,931     |
| fineweb_edu_english | 100,001,720       |
| fineweb_english     | 100,000,727       |
| fineweb2_german     | 600,000,024       |
| All                 | 2,000,026,571     |

## Pretraining Data (30-50-5-5-10 Mix)

We want to use 200B subtokens (measured with our previously trained tokenizer) as our vocab corpus from the following datasets:

| Dataset               | Percentage | Tokens   | Language |
|-----------------------|------------|---------:|----------|
| FineWeb2 (German)     | 30%        |  60B     | German   |
| FinePdfs (German)     | 50%        | 100B     | German   |
| FineWeb (English)     | 5%         |  10B     | English  |
| FineWeb-Edu (English) | 5%         |  10B     | English  |
| FinePdfs (English)    | 10%        |  20B     | English  |

For creating the pretraining data, we use an own way for data preprocessing (so no longer the DataTrove pipeline):

DataTrove cannot handle termination after n subtokens, when using it with multiple tasks/workers. The `TokenLimiter` perfectly works when using `tasks=1` and `workers=1`, but when using it with multiple workers/tasks then the `max_tokens` limit is applied to each chunk that is read in parallel.

In our final solution we make use of Dataset streaming. This also heavily reduced the used and needed storage space.

Additionally, we switch from using TensorFlow Datasets. One reason is: to build the final "Dataset" all different corpora must be located on disk. It is not possible to combine e.g. preprocessed corpora after another. Using Grain and Array Records, we can upload preprocessed datasets (e.g FineWeb English) after another and combine it later for pretraining via configuration. Technically, it is also possible to define datasets mixings on a configuration file level later. So using Grain and Array Records has a lot of more advantages over TensorFlow Datasets or Hugging Face Datasets in MaxText.

The dataset creation can be started with:

```bash
# FinePdfs (English)
python3 -m pipeline.build_array_record_dataset --config pipeline/configs/30-50-5-5-10-mix/finepdfs_english.yaml

# FineWeb (English)
python3 -m pipeline.build_array_record_dataset --config pipeline/configs/30-50-5-5-10-mix/fineweb_english.yaml
```
