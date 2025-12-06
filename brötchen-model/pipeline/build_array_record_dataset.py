import argparse
import os
import yaml

from array_record.python.array_record_module import ArrayRecordWriter
from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer

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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_after_n_documents = config["array_record"]["safe_after_n_documents"]

    for dataset_name, dataset_config in config['datasets'].items():
        print(dataset_name, dataset_config)

        hf_data_files = dataset_config["hf_data_files"]
        dataset_tokenizer = config["tokenizer"]["name"]
        dataset_max_subtokens = dataset_config["max_subtokens"]
        dataset_hf_identifier = dataset_config["hf_identifier"]
        dataset_output_path = Path(dataset_config["output_path"])
        tokenizer = AutoTokenizer.from_pretrained(dataset_tokenizer)

        # For our record array writer
        dataset_output_path.mkdir(parents=True, exist_ok=True)

        # Some stats
        total_subtokens = 0
        processed_docs = 0

        dataset = load_dataset(
            dataset_hf_identifier,
            data_files=hf_data_files,
            split="train",
            streaming=True
        )

        current_shard_index = 0

        shard_path = (dataset_output_path / f"train_{current_shard_index:06d}.arecord").absolute().as_posix()

        writer =  ArrayRecordWriter(shard_path, "group_size:1")

        try:
            for example in dataset:
                text = example['text']

                # Tokenize the document - try with add_special_tokens=False
                tokens = tokenizer.encode(text, add_special_tokens=False)
                num_tokens = len(tokens)
                
                # Update counter
                total_subtokens += num_tokens
                processed_docs += 1

                if processed_docs % 1_000 == 0:
                    print(f"Processed {processed_docs:,} documents | Total subtokens: {total_subtokens:,}")

                # Log progress
                if processed_docs % save_after_n_documents == 0:
                    print(f"Processed {processed_docs:,} documents | Total subtokens: {total_subtokens:,} | Written Array Record Shard: {current_shard_index:,}")
                    # Close current writer
                    writer.close()

                    # Prepare new writer
                    current_shard_index += 1
                    shard_path = (dataset_output_path / f"train_{current_shard_index:06d}.arecord").absolute().as_posix()

                    writer =  ArrayRecordWriter(shard_path, "group_size:1")

                writer.write(str.encode(text))

                # Check if threshold is reached
                if total_subtokens >= dataset_max_subtokens:
                    print(f"\nThreshold reached!")
                    print(f"Total subtokens: {total_subtokens:,}")
                    print(f"Total documents: {processed_docs:,}")
                    writer.close()
                    break
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            print(f"\nFinal stats for {dataset_name}:")
            print(f"Documents processed: {processed_docs:,}")
            print(f"Total subtokens: {total_subtokens:,}")
            writer.close()
        
        # Explicitly delete tokenizer to help with cleanup
        del tokenizer

if __name__ == "__main__":
    main()
