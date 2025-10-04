# %%
%load_ext autoreload
%autoreload 2
# %%

# %%
"""Interactive LatentQA dataset exploration using Python script notebook cells."""

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from lit.configs.train_config import train_config
from lit.utils.infra_utils import update_config, get_tokenizer
from lit.utils.dataset_utils import (
    DataCollatorForLatentQA,
    get_batch_sampler,
    get_dataset,
)

# %%
DEFAULT_OVERRIDES: dict[str, Any] = {
    "target_model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "train_stimulus_completion": "data/train/stimulus_completion.json",
    "train_stimulus": "data/train/stimulus.json",
    "train_control": "data/train/control.json",
    "train_qa": "data/train/qa.json",
    # Uncomment the lines below if you have evaluation splits configured
    # "eval_stimulus_completion": "data/eval/stimulus_completion.json",
    # "eval_stimulus": "data/eval/stimulus.json",
    # "eval_control": "data/eval/control.json",
    # "eval_qa": "data/eval/qa.json",
}


def build_args(overrides: dict[str, Any] | None = None) -> train_config:
    """Return a populated train_config with notebook-friendly overrides."""
    args = train_config()
    merged = {**DEFAULT_OVERRIDES, **(overrides or {})}
    update_config(args, **merged)
    return args


def load_components(
    mode: str = "train",
    overrides: dict[str, Any] | None = None,
    *,
    num_workers: int | None = None,
) -> tuple[Any, DataLoader]:
    """Build dataset and DataLoader for the requested mode."""
    if mode not in {"train", "eval", "validation", "val"}:
        raise ValueError("mode must be one of {'train', 'eval', 'validation', 'val'}")
    args = build_args(overrides)
    if num_workers is not None:
        args.num_workers_dataloader = num_workers
    is_train = mode == "train"

    tokenizer = get_tokenizer(args.target_model_name)
    dataset = get_dataset(args, tokenizer, train=is_train)
    collator = DataCollatorForLatentQA(
        tokenizer,
        mask_all_but_last=False,
        nudge_persona=args.nudge_persona,
        modify_chat_template=args.modify_chat_template,
    )
    sampler_mode = "train" if is_train else "val"
    sampler = get_batch_sampler(dataset, args, sampler_mode)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers_dataloader,
        pin_memory=False,
    )
    return dataset, dataloader


def preview_dataset(
    dataset,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return a few raw samples to inspect interactively."""
    results: list[dict[str, Any]] = []
    for idx in range(min(limit, len(dataset))):
        sample = dataset[idx]
        print(f"\n\n\n\nIDX {idx}")
        for key, value in sample.items():
            print(f"\n\n{key}: {value}")
        results.append(
            {
                "index": idx,
                "mask_type": sample["mask_type"],
                "dialog_roles": [turn["role"] for turn in sample["dialog"]],
                "read_prompt_excerpt": sample["read_prompt"][:120],
            }
        )
    return results


def summarize_batch(batch: dict[str, Any]) -> list[str]:
    """Produce human-readable descriptions for tensors in a batch."""
    summary: list[str] = []
    for key, value in batch.items():
        if torch.is_tensor(value):
            summary.append(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
            continue
        if hasattr(value, "items"):
            for inner_key, inner_value in value.items():
                if torch.is_tensor(inner_value):
                    summary.append(f"{key}.{inner_key}: shape={tuple(inner_value.shape)} dtype={inner_value.dtype}")
                else:
                    summary.append(f"{key}.{inner_key}: {type(inner_value).__name__}")
            continue
        summary.append(f"{key}: {type(value).__name__}")
    return summary


def save_batch(batch: dict[str, Any], path: str | Path) -> Path:
    """Persist a batch for offline inspection."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, target)
    return target


# %%
# Example usage (uncomment and run in notebooks):

tokenizer = get_tokenizer(DEFAULT_OVERRIDES["target_model_name"])

dataset, dataloader = load_components(mode="train")
# %%
preview_dataset(dataset, limit=10)

# %%

print(dataset[0])


# %%

read_lengths = set()
write_lengths = set()

for dp in dataset:
    read_lengths.add(len(dp["read_prompt"]))
    write_lengths.add(len(dp["dialog"]))

print(read_lengths, write_lengths)

# %%
raise ValueError
# %%
first_batch = next(iter(dataloader))

# %%
summarize_batch(first_batch)

# %%

print(summarize_batch(first_batch))
print(first_batch.keys())

# %%

print(tokenizer.batch_decode(first_batch["tokenized_read"]["input_ids"]))

for i in range(first_batch["tokenized_read"]["input_ids"].shape[0]):
    print(f"Prompt {i}: {tokenizer.decode(first_batch['tokenized_read']['input_ids'][i])}")
    print(f"Prompt {i}: {tokenizer.decode(first_batch['tokenized_write']['input_ids'][i])}")


print(f"{first_batch['tokenized_read']['input_ids']=}")
print(f"{first_batch['tokenized_read']['input_ids'].shape=}")
print(f"{first_batch['tokenized_write']['input_ids']=}")
print(f"{first_batch['tokenized_write']['labels']=}")
print(f"{first_batch['tokenized_write']['input_ids'].shape=}")
# print(f'{first_batch["tokenized_read"]["attention_mask"]=}')
# print(f'{first_batch["tokenized_read"]["attention_mask"].shape=}')
print(f"{first_batch['tokenized_read']['input_ids'].shape=}")
print(first_batch["read_lengths"])
print(first_batch["verb_lengths"])
print(first_batch["write_lengths"])

# %%


batch_idx = 1
verb_length = first_batch["verb_lengths"][batch_idx]
length = first_batch["tokenized_read"]["input_ids"].shape[1]
padding = length - first_batch["read_lengths"][batch_idx]


verb_length += padding

print(f"{first_batch['tokenized_read']['input_ids'][batch_idx, verb_length:]}")

print(tokenizer.decode(first_batch["tokenized_read"]["input_ids"][batch_idx, verb_length:]))
# %%

# %%
save_batch(first_batch, "out/debug/first_batch.pt")


# %%
