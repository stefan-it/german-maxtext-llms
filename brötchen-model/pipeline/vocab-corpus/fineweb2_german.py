from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

from ..token_limiter import TokenLimiter


tokenizer = "german-maxtext-slms/nano-neox-tokenizer"
suffix = "fineweb2_german"
download_folder = f"./{suffix}/data/deu_Latn/train"
max_tokens = 600_000_000
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
