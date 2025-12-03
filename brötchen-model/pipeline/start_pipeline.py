import argparse
import yaml

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter

from .token_limiter import TokenLimiter


def main():
    parser = argparse.ArgumentParser(
        description="DataTrove Pipeline for creating SLM Datasets"
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to yaml configuration file'
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for dataset_name, dataset_config in config['datasets'].items():
        print(dataset_name, dataset_config)

        dataset_path = dataset_config["path"]
        dataset_tokenizer = config["tokenizer"]["name"]
        dataset_max_tokens = dataset_config["max_tokens"]
        dataset_output_path = dataset_config["output_path"]

        pipeline = [
            ParquetReader(data_folder=f"{dataset_name}/{dataset_path}"),
            TokenLimiter(tokenizer_name_or_path=dataset_tokenizer, max_tokens=dataset_max_tokens),
            ParquetWriter(output_folder=f"{dataset_output_path}/{dataset_name}",compression=None),
        ]

        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=config["datatrove"]["tasks"],
            workers=config["datatrove"]["workers"],
            logging_dir="./logs",
        )

        executor.run()

if __name__ == "__main__":
    main()