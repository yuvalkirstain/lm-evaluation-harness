import json
import math
import os
import random
import tempfile
from datasets import load_dataset
from torch.utils.data import DataLoader

from lm_eval.pl_models import (
    LitGPT2,
    LitT5, VAL_LOSS, VAL_ACC
)

from transformers import set_seed
import pytorch_lightning as pl

from dataclasses import (
    dataclass,
    asdict
)


@dataclass
class TrainArgs:
    train_set_size: int
    task_name: str
    model_type: str
    gradient_clip_val: int
    weight_decay: float
    learning_rate: float
    optimizer_type: str
    lr_scheduler_type: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    save_prefix: str = None
    dropout: float = None
    pretrained: str = None
    device: str = None
    min_train_steps: int = None
    num_train_epochs: int = 10
    verbose: str = ""
    per_device_eval_batch_size: int = 1
    preprocessing_num_workers: int = 1
    overwrite_cache: bool = True
    min_warmup_steps: int = 100
    num_warmup_steps: int = None
    warmup_ratio: float = 0.1
    seed: int = 1234
    max_train_steps: int = None
    monitor: str = None
    monitor_mode: str = None


def write_datasets_fo_read(train_set, save_prefix):
    num_in_train = max((len(train_set) * 3) // 4, len(train_set) - 400)
    train_set, dev_set = train_set[:num_in_train], train_set[num_in_train:]
    temp_dir = tempfile.TemporaryDirectory(prefix=save_prefix)
    print(f"saving stuff in {temp_dir}")
    train_file = f"{temp_dir.name}/train.json"
    dev_file = train_file.replace("train", "dev")
    with open(train_file, "w") as f:
        for dp in train_set:
            f.write(json.dumps(dp) + '\n')
    with open(dev_file, "w") as f:
        for dp in dev_set:
            f.write(json.dumps(dp) + '\n')
    return train_file, dev_file, temp_dir


def load_raw_datasests(train_set, save_prefix):
    train_file, dev_file, temp_dir = write_datasets_fo_read(train_set, save_prefix)
    extension = train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files={"train": train_file, "validation": dev_file})

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        print(f"Sample {index} of the training set: {raw_datasets['train'][index]}.")

    return raw_datasets, temp_dir


def get_monitor_name(task_name):
    if task_name in ["arc_easy", "copa", "openbookqa", "lambada_cloze", "triviaqa", "piqa", "webqs", "nq_open", "winogrande", "race"]:
        return VAL_LOSS, "min"
    elif task_name in ["rte", "sst", "wic", "multirc", "anli_r1", "wsc", "boolq", "squad2", "squad1", "drop"]:
        return VAL_ACC, "max"
    else:
        raise ValueError(f"We don't support task {task_name}")


def train_lm(model, tokenizer, train_set, task_name, train_args):

    train_args = TrainArgs(train_set_size=len(train_set), task_name=task_name, **train_args)

    # If passed along, set the training seed now.
    set_seed(train_args.seed)

    logger = pl.loggers.CometLogger(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name=os.environ.get('COMET_PROJECT', "few-shot"),
        workspace=os.environ.get('COMET_WORKSPACE', "yuvalkirstain"),
        # save_dir=temp_dir.name,
        # offline=True
    )

    raw_datasets, temp_dir = load_raw_datasests(train_set, train_args.save_prefix)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    train_args.monitor, train_args.monitor_mode = get_monitor_name(task_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=train_args.monitor,
        dirpath=os.path.join(temp_dir.name, "checkpoints"),
        save_top_k=1,
        mode=train_args.monitor_mode,
    )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Scheduler and math around the number of training steps.
    # TODO not sure if this is accurate for multi-gpu
    num_update_steps_per_epoch = math.ceil(len(raw_datasets["train"]) / (train_args.gradient_accumulation_steps * train_args.per_device_train_batch_size))
    if train_args.num_train_epochs * num_update_steps_per_epoch < train_args.min_train_steps and train_args.max_train_steps is None:
        train_args.max_train_steps = train_args.min_train_steps
    if train_args.max_train_steps is None:
        train_args.max_train_steps = train_args.num_train_epochs * num_update_steps_per_epoch
    else:
        train_args.num_train_epochs = math.ceil(train_args.max_train_steps / num_update_steps_per_epoch)

    train_args.num_warmup_steps = max(train_args.min_warmup_steps, int(train_args.max_train_steps * train_args.warmup_ratio))

    model_cls = LitGPT2 if train_args.model_type == "gpt2" else LitT5
    model_finetuner = model_cls(model,
                                train_args.lr_scheduler_type,
                                train_args.num_warmup_steps,
                                train_args.max_train_steps,
                                train_args.weight_decay,
                                train_args.learning_rate,
                                tokenizer)

    tokenized_datasets = raw_datasets.map(
        model_finetuner.tokenize_function,
        batched=True,
        num_proc=train_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not train_args.overwrite_cache,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=model_finetuner.collate_fn, batch_size=train_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=model_finetuner.collate_fn, batch_size=train_args.per_device_eval_batch_size
    )

    logger.log_hyperparams(asdict(train_args))
    print(train_args)
    trainer = pl.Trainer(
        gpus=-1,
        accumulate_grad_batches=train_args.gradient_accumulation_steps,
        plugins=None,
        precision=32,
        logger=logger,
        max_steps=train_args.max_train_steps,
        min_steps=train_args.min_train_steps,
        gradient_clip_val=train_args.gradient_clip_val,
        callbacks=[checkpoint_callback, lr_monitor],
        # log_every_n_steps=1,
        # flush_logs_every_n_steps=1
    )
    trainer.fit(model_finetuner, train_dataloader, eval_dataloader)
    print(f"best checkpoint is: {checkpoint_callback.best_model_path}")
    test_model = model_cls.load_from_checkpoint(checkpoint_callback.best_model_path)

    train_args = asdict(train_args)
    train_args["best_model_path"] = checkpoint_callback.best_model_path
    train_args["best_model_score"] = checkpoint_callback.best_model_score.item()
    train_args["previous_experiment"] = logger.experiment.get_key()
    temp_dir.cleanup()
    return test_model.model.eval(), train_args
