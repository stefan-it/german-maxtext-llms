import argparse
import pandas as pd
from tabulate import tabulate
import yaml

from pathlib import Path


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

    dataset_name_stats = {}
    headers = ["Dataset name", "Total Subtokens"]
    rows = []

    for dataset_name, dataset_config in config['datasets'].items():
        #print(dataset_name, dataset_config)
        dataset_output_path = Path(dataset_config["output_path"]) / dataset_name

        total_subtokens = 0

        if not dataset_output_path.exists():
            continue

        for parquet_file in dataset_output_path.iterdir():
            if not parquet_file.name.endswith(".parquet"):
                continue

            df = pd.read_parquet(parquet_file)
            total_subtokens += df['metadata'].apply(lambda x: x.get('subtokens', 0)).sum()
        
        dataset_name_stats[dataset_name] = total_subtokens

        rows.append([dataset_name, f"{total_subtokens:,}"])

    # Add total row
    rows.append(["All", f"{sum(dataset_name_stats.values()):,}"])

    print(tabulate(rows, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
