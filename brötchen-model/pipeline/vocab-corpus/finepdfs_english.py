from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

from ..token_limiter import TokenLimiter


tokenizer = "german-maxtext-slms/nano-neox-tokenizer"
suffix = "finepdfs_english"
download_folder = f"./{suffix}/data/eng_Latn/train"
max_tokens = 200_000_000
output_folder = f"./parquet_{suffix}"

pipeline = [
    ParquetReader(data_folder=download_folder),
    TokenLimiter(tokenizer_name_or_path=tokenizer, max_tokens=max_tokens),
    ParquetWriter(output_folder=output_folder,compression="snappy"),
]

executor = LocalPipelineExecutor(
    pipeline=pipeline,
    tasks=20,
    logging_dir="./logs",
)

executor.run()
