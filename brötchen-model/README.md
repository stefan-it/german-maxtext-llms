# Brötchen SLM

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
