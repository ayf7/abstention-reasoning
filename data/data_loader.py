from pathlib import Path

from datasets import load_dataset, concatenate_datasets


BASE_DIR = Path(__file__).resolve().parent


DATASETS = {
    "knights_and_knaves": {
        "hf_id": "K-and-K/perturbed-knights-and-knaves",
        "configs": ["train", "test"],
    },
    "connections": {
        "hf_id": "tm21cy/NYT-Connections",
        "configs": None,
    },
    "connections_synthetic": {
        "hf_id": "Asap7772/NYT-Connections-Synthetic-Better-o4mini-all",
        "configs": None,
    },
    "countdown": {
        "hf_id": "Jiayi-Pan/Countdown-Tasks-3to4",
        "configs": None,
    },
}


def download_and_save(dataset_name: str, target_subdir: str) -> None:
    """
    Download a Hugging Face dataset and save all splits as JSONL.

    Files are written under:
        data/datasets/<target_subdir>/<split>.jsonl
    """
    target_dir = BASE_DIR / target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_dataset(dataset_name)

    # Save each split (or sub-dataset) to its own JSONL file.
    for split_name, split_dataset in dataset_dict.items():
        out_path = target_dir / f"{split_name}.jsonl"
        split_dataset.to_json(out_path)
        print(f"Saved {dataset_name!r} ({split_name}) to {out_path}")


def download_and_save_configs_combined(
    dataset_name: str, target_subdir: str, configs: list[str]
) -> None:
    """
    For datasets that expose configs like 'train'/'test', load each config,
    combine all of its internal splits, and save one file per config:

        data/datasets/<target_subdir>/train.jsonl
        data/datasets/<target_subdir>/train.parquet
        ...
    """
    target_dir = BASE_DIR / target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    for config_name in configs:
        dataset_dict = load_dataset(dataset_name, config_name)
        combined = concatenate_datasets(list(dataset_dict.values()))

        json_path = target_dir / f"{config_name}.jsonl"
        # parquet_path = target_dir / f"{config_name}.parquet"

        combined.to_json(json_path)
        # combined.to_parquet(parquet_path)

        print(
            f"Saved combined config {config_name!r} of {dataset_name!r} "
            f"to {json_path}"
        )


def main() -> None:
    for local_dir, spec in DATASETS.items():
        hf_id = spec["hf_id"]
        configs = spec["configs"]

        if configs:
            # Dataset uses explicit configs (e.g. 'train', 'test').
            download_and_save_configs_combined(hf_id, local_dir, configs=configs)
        else:
            # Dataset exposes standard splits like 'train', 'test'.
            download_and_save(hf_id, local_dir)


if __name__ == "__main__":
    main()
