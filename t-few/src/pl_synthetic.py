import os
import torch
import argparse
import json
import pickle
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_from_disk
from tqdm import tqdm

from src.data import FinetuneDataModule, get_dataset_reader
from src.models.EncoderDecoder import EncoderDecoder
from src.models.modify_model import modify_transformer
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds


def get_transformer(config):
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)
    tokenizer.model_max_length = config.max_seq_len
    model = modify_transformer(model, config)
    return tokenizer, model


def main(config):
    print("Setting up dataset reader and tokenizer...")
    config.few_shot = True
    tokenizer, model = get_transformer(config)
    dataset_reader = get_dataset_reader(config)

    print("Preparing data module...")
    datamodule = FinetuneDataModule(config, tokenizer, dataset_reader)
    datamodule.setup(stage="fit")

    print("Building model and trainer...")
    val_collate_fn = datamodule.val_dataloader().collate_fn
    datamodule.val_dataloader = lambda: None  # Disable validation dataloader for inference
    model = EncoderDecoder(config, tokenizer, model, dataset_reader)

    logger = TensorBoardLogger(config.exp_dir, name="log")
    trainer = Trainer(
        enable_checkpointing=False,
        gpus=torch.cuda.device_count(),
        precision=config.compute_precision,
        amp_backend="native",
        strategy=config.compute_strategy if config.compute_strategy != "none" else None,
        logger=logger,
        log_every_n_steps=4,
        max_steps=config.num_steps,
        min_steps=config.num_steps,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=config.eval_epoch_interval,
        accumulate_grad_batches=config.grad_accum_factor,
        gradient_clip_val=config.grad_clip_norm,
    )

    print("Starting training...")
    trainer.fit(model, datamodule)
    print("Training complete.")

    # Inference on synthetic data
    print("Loading synthetic dataset...")
    synth_data_path = os.path.join(config.datasets_offline, config.dataset + '_synthetic')
    synth_data = load_from_disk(synth_data_path)

    if "idx" not in synth_data.column_names:
        synth_data = synth_data.map(lambda ex, idx: {**ex, "idx": idx}, with_indices=True)

    print("Preparing synthetic data loader...")
    synth_dataset = datamodule.dev_dataset
    synth_dataset.dataset = synth_data
    synth_loader = torch.utils.data.DataLoader(
        synth_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=val_collate_fn,
        num_workers=min([config.eval_batch_size, config.num_workers]),
    )

    print("Running inference on synthetic data...")
    model.eval()
    all_outputs = []

    for batch in tqdm(synth_loader, desc="Inferencing on synthetic data", leave=False):
        with torch.no_grad():
            output = model.predict(batch)
            if isinstance(output, list):
                all_outputs.extend(output)
            else:
                all_outputs.append(output)

    print("Saving synthetic inference results...")
    out_dir = config.exp_dir.replace(config.dataset, config.dataset + '_synthetic')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 't0.p')
    # Merge all keys into flat lists
    merged = {}
    for entry in all_outputs:
        for key, value in entry.items():
            if key not in merged:
                merged[key] = []
            if isinstance(value, list):
                merged[key].extend(value)
            else:
                merged[key].append(value)
    with open(out_path, 'wb') as f:
        pickle.dump(merged, f)
    print(f"Saved synthetic inference results to {out_path}")

    # Inference on test data
    print("Loading test dataset...")
    test_data_path = os.path.join(config.datasets_offline, config.dataset + '_test')
    test_data = load_from_disk(test_data_path)

    if "idx" not in test_data.column_names:
        test_data = test_data.map(lambda ex, idx: {**ex, "idx": idx}, with_indices=True)

    print("Preparing test data loader...")
    test_dataset = datamodule.dev_dataset
    test_dataset.dataset = test_data
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=val_collate_fn,
        num_workers=min([config.eval_batch_size, config.num_workers]),
    )

    print("Running inference on test data...")
    all_test_outputs = []

    for batch in tqdm(test_loader, desc="Inferencing on test data", leave=False):
        with torch.no_grad():
            output = model.predict(batch)
            if isinstance(output, list):
                all_test_outputs.extend(output)
            else:
                all_test_outputs.append(output)

    print("Saving test inference results...")
    test_out_dir = config.exp_dir.replace(config.dataset, config.dataset + '_test')
    os.makedirs(test_out_dir, exist_ok=True)
    test_out_path = os.path.join(test_out_dir, 't0.p')
    # Merge all keys into flat lists
    test_merged = {}
    for entry in all_test_outputs:
        for key, value in entry.items():
            if key not in test_merged:
                test_merged[key] = []
            if isinstance(value, list):
                test_merged[key].extend(value)
            else:
                test_merged[key].append(value)
    with open(test_out_path, 'wb') as f:
        pickle.dump(test_merged, f)
    print(f"Saved test inference results to {test_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)
    config.datasets_offline = "/hdd/hans/TabLLM/datasets_serialized"
    out_dir = config.exp_dir.replace(config.dataset, config.dataset + '_synthetic')
    test_out_dir = config.exp_dir.replace(config.dataset, config.dataset + '_test')
    if os.path.exists(out_dir) and os.path.exists(test_out_dir):
        print(f"Output directories {out_dir} and {test_out_dir} already exist. Skipping run.")
    else:
        print(f"Start synthetic distillation experiment {config.exp_name}")
        set_seeds(config.seed)
        main(config)
