from typing import List, Dict, Union

import torch
import pytorch_lightning as pl
from lm_eval.linear_scheduler_with_warmup import LinearSchedulerWithWarmup
from fairscale.nn import checkpoint_wrapper, auto_wrap

from transformers import (
    AdamW,
    Adafactor,
    get_scheduler,
)

VAL_ACC = "valid/acc_epoch"
VAL_LOSS = "valid/loss_epoch"

class LitGPT2(pl.LightningModule):
    def __init__(self,
                 model,
                 lr_scheduler_type,
                 num_warmup_steps,
                 max_train_steps,
                 weight_decay,
                 learning_rate,
                 tokenizer):
        super().__init__()
        self.model = model
        self.lr_scheduler_type = lr_scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.max_train_steps = max_train_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.tokenizer = tokenizer
        self.save_hyperparameters()

    def configure_sharded_model(self):
        self.model = self.model

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        self.model.parallelize()
        outputs = self.model(**batch, return_dict=False)
        return outputs

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer):
        current_lr = [d['lr'] for d in optimizer.param_groups][0]
        self.log('train/lr', current_lr, on_step=True, sync_dist=True, logger=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0]
        if torch.isnan(loss).any():
            raise ValueError("got nan!")
        self.log('train/loss', loss, on_step=True, on_epoch=True, sync_dist=True, logger=True, prog_bar=True)
        self.log('train/global_step', self.global_step, on_step=True, sync_dist=True, logger=True, prog_bar=True)
        self.log('train/batch_size', len(outputs[1]), on_step=True, sync_dist=True, logger=True, prog_bar=True)
        return loss

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def validation_step(self, batch, batch_nb):
        outputs = self(batch)
        loss, logits = outputs[0], outputs[1]
        logits = logits[:, :-1]
        labels = batch["labels"][:, 1:]
        greedy_tokens = logits.argmax(dim=-1)
        # TODO(Yuval) - this is very strict acc, perhaps change
        batch_acc = torch.zeros(labels.size(0)).type_as(logits)
        for idx in range(labels.size(0)):
            relevant_indices = (labels[idx] != -100).float().nonzero()[:, 0]
            relevant_labels = labels[idx][relevant_indices]
            relevant_greedy_tokens = greedy_tokens[idx][relevant_indices]
            cur_acc = (relevant_labels == relevant_greedy_tokens).all(dim=-1).float()
            batch_acc[idx] = cur_acc
        return {"val_loss": loss, "val_acc": batch_acc.mean()}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log(VAL_LOSS, loss, prog_bar=True, sync_dist=True, logger=True)  # default on val/test is on_epoch only
        self.log(VAL_ACC, val_acc, prog_bar=True, sync_dist=True, logger=True)
        return {"val_loss": loss, "val_acc": val_acc}

    def configure_optimizers(self):
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)

        lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )

        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
        }]

    def collate_fn(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(dp["input_ids"]) for dp in batch)
        input_ids = [dp["input_ids"] + [self.tokenizer.pad_token_id] * (max_length - len(dp["input_ids"])) for dp in
                     batch]
        attention_mask = [dp["attention_mask"] + [0] * (max_length - len(dp["attention_mask"])) for dp in batch]
        labels = [dp["labels"] + [-100] * (max_length - len(dp["labels"])) for dp in batch]
        return {"input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels)}

    def tokenize_function(self, examples):
        examples["context"] = [context if context else self.tokenizer.eos_token for context in examples["context"]]
        full_texts = [context + completion for context, completion in
                      zip(examples["context"], examples["completion"])]
        context_ids = self.tokenizer(examples["context"])["input_ids"]
        data = self.tokenizer(full_texts)
        data["labels"] = data["input_ids"].copy()
        for i, (label, context) in enumerate(zip(data["labels"], context_ids)):
            data["labels"][i] = [-100] * len(context) + label[len(context):]
            for k in data:
                data[k][i] = data[k][i][-self.tokenizer.model_max_length:]
        return data


class LitT5(LitGPT2):
    def validation_step(self, batch, batch_nb):
        outputs = self(batch)
        loss, logits = outputs[0], outputs[1]
        labels = batch["labels"]
        greedy_tokens = logits.argmax(dim=-1)
        # TODO(Yuval) - this is very strict acc, perhaps change
        batch_acc = torch.zeros(labels.size(0)).type_as(logits)
        for idx in range(labels.size(0)):
            relevant_indices = (labels[idx] != -100).float().nonzero()[:, 0]
            relevant_labels = labels[idx][relevant_indices]
            relevant_greedy_tokens = greedy_tokens[idx][relevant_indices]
            cur_acc = (relevant_labels == relevant_greedy_tokens).all(dim=-1).float()
            batch_acc[idx] = cur_acc
        return {"val_loss": loss, "val_acc": batch_acc.mean()}

    def configure_optimizers(self):
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False,
                              warmup_init=False,
                              lr=self.learning_rate)

        # lr_scheduler = get_scheduler(
        #     name=self.lr_scheduler_type,
        #     optimizer=optimizer,
        #     num_warmup_steps=self.num_warmup_steps,
        #     num_training_steps=self.max_train_steps,
        # )
        assert self.lr_scheduler_type == "linear", "We currently only support linear"
        lr_scheduler = LinearSchedulerWithWarmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )

        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
        }]

    def collate_fn(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(dp["input_ids"]) for dp in batch)
        input_ids = [dp["input_ids"] + [self.tokenizer.pad_token_id] * (max_length - len(dp["input_ids"])) for dp in
                     batch]
        attention_mask = [dp["attention_mask"] + [0] * (max_length - len(dp["input_ids"])) for dp in batch]
        labels_max_length = max(len(dp["labels"]) for dp in batch)
        labels = [dp["labels"] + [-100] * (labels_max_length - len(dp["labels"])) for dp in batch]
        return {"input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels)}

    def tokenize_function(self, examples):
        full_texts = [context.strip() + "<extra_id_0>." if "<extra_id_0>" not in context else context for context in
                      examples["context"]]
        data = self.tokenizer(full_texts)
        data["labels"] = \
            self.tokenizer(
                ["<extra_id_0> " + completion.strip() + "<extra_id_1>" for completion in examples["completion"]],
                add_special_tokens=False)["input_ids"]
        for k in data:
            for i in range(len(data[k])):
                data[k][i] = data[k][i][-self.tokenizer.model_max_length:]
        return data
