"""
create_data_interactive_v2.py

Interactive comparison of the original LatentQA dataset vs. the standalone loader.

What this does:
- Loads dataset via the original code path (lit.utils.dataset_utils.get_dataset)
  without requiring a real tokenizer (uses a tiny DummyTokenizer).
- Loads dataset via the standalone loader (latentqa_dataset_standalone.load_latentqa_dataset).
- Compares that each index returns the same core dict fields:
  {"read_prompt", "dialog", "mask_type"}.
- Prints a short summary and first mismatch if any.

Run:
  python latentqa/create_data_interactive_v2.py

You can edit DEFAULT_OVERRIDES below to point to your JSON files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    # When executed as part of the latentqa package
    from .lit.configs.train_config import train_config
    from .lit.utils.dataset_utils import get_dataset
except ImportError:  # Fallback for direct script execution
    from lit.configs.train_config import train_config
    from lit.utils.dataset_utils import get_dataset

try:
    # Prefer package-relative import when running as `python -m latentqa.create_data_interactive_v2`
    from . import latentqa_dataset_standalone as standalone
except Exception:
    # Fallback for direct script execution from repo root
    import latentqa_dataset_standalone as standalone


# --------------------------------------------------------------------------------------
# Defaults (edit these or pass your own via code changes)
# --------------------------------------------------------------------------------------


DEFAULT_OVERRIDES = {
    "target_model_name": "meta-llama/Llama-3.1-8B-Instruct",
    # For a lightweight comparison by default, use just stimulus_completion + qa.
    # You can also set the others if you have them.
    "train_system": "",
    "train_stimulus_completion": "data/train/stimulus_completion.json",
    "train_stimulus": "data/train/stimulus.json",
    "train_control": "data/train/control.json",
    "train_qa": "data/train/qa.json",
    # Optional controls; keep deterministic
    "filter": "",
    "train_percent": 1.0,
    "add_thought_tokens": False,
    "seed": 42,
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _resolve_path(base: Path, p: str | None) -> str:
    if not p:
        return ""
    path = Path(p)
    return str(path if path.is_absolute() else (base / path))


def _build_args(overrides: Optional[Dict[str, Any]] = None) -> train_config:
    args = train_config()
    if overrides:
        for k, v in overrides.items():
            # Only set attrs that actually exist on the dataclass
            if hasattr(args, k):
                setattr(args, k, v)
    return args


@dataclass
class DummyTokenizer:
    """Minimal shim to satisfy LatentQADataset constructor.

    The original dataset only probes `tokenizer.name_or_path` to select a chat
    template, but does not tokenize because that code is commented out.
    """

    name_or_path: str


def load_original_dataset(*, overrides: Optional[Dict[str, Any]] = None):
    base = Path(__file__).resolve().parent  # latentqa/ directory
    merged = {**DEFAULT_OVERRIDES, **(overrides or {})}

    # For deterministic cross-path comparison, enforce full dataset order.
    assert merged["train_percent"] == 1.0

    # Resolve file paths to absolute to be robust to CWD
    for key in [
        "train_system",
        "train_stimulus_completion",
        "train_stimulus",
        "train_control",
        "train_qa",
    ]:
        if key in merged:
            merged[key] = _resolve_path(base, merged[key])

    args = _build_args(merged)
    # Provide a lightweight tokenizer substitute
    tok = DummyTokenizer(name_or_path=args.target_model_name)
    dataset = get_dataset(args, tokenizer=tok, train=True)
    return dataset, args


def load_standalone_dataset(args: train_config):
    paths = standalone.DataPaths(
        system=args.train_system,
        stimulus_completion=args.train_stimulus_completion,
        stimulus=args.train_stimulus,
        control=args.train_control,
        qa=args.train_qa,
    )
    ds = standalone.load_latentqa_dataset(
        paths,
        filter_prefixes=(args.filter.split("-") if args.filter else []),
        train_percent=args.train_percent,
        add_thought_tokens=args.add_thought_tokens,
        seed=args.seed,
    )
    return ds


def _project_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "read_prompt": item["read_prompt"],
        "dialog": item["dialog"],
        "mask_type": item["mask_type"],
    }


def _equal_core(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return _project_item(a) == _project_item(b)


def compare_datasets(ds_a, ds_b, *, show_first_diff: bool = True) -> Tuple[int, int, int]:
    len_a, len_b = len(ds_a), len(ds_b)
    n = min(len_a, len_b)
    mismatches = []
    for i in range(n):
        if not _equal_core(ds_a[i], ds_b[i]):
            mismatches.append(i)
            if show_first_diff and len(mismatches) == 1:
                print(f"\nFirst mismatch at index {i}")
                print("Original (projected):")
                print(json.dumps(_project_item(ds_a[i]), ensure_ascii=False, indent=2))
                print("Standalone (projected):")
                print(json.dumps(_project_item(ds_b[i]), ensure_ascii=False, indent=2))
    if len_a != len_b:
        print(f"\nLength mismatch: original={len_a}, standalone={len_b}")
    return len_a, len_b, len(mismatches)


def main():
    ds_orig, args = load_original_dataset()
    ds_standalone = load_standalone_dataset(args)

    print("Loaded datasets:")
    print(f"  original:   {len(ds_orig)} items")
    print(f"  standalone: {len(ds_standalone)} items")

    len_a, len_b, num_diff = compare_datasets(ds_orig, ds_standalone)
    print("\nComparison summary:")
    print(f"  lengths: original={len_a}, standalone={len_b}")
    print(f"  mismatches (core fields): {num_diff}")

    # Assert shape for stimulus_completion: roles must be [user, assistant, user, assistant]
    expected_roles = ["user", "assistant", "user", "assistant"]

    # Check standalone path using the 'source' field
    sc_count = 0
    for i in range(len(ds_standalone)):
        item = ds_standalone[i]
        if item.get("source") == "stimulus_completion":
            roles = [t["role"] for t in item["read_prompt"]]
            assert roles == expected_roles, f"Standalone 'stimulus_completion' roles mismatch at idx {i}: {roles}"
            sc_count += 1
    if sc_count == 0:
        print("[warn] No 'stimulus_completion' items found in standalone dataset to verify.")

    # Check original path by grouping via id_tuples lengths; group 1 = stimulus_completion
    if hasattr(ds_orig, "id_tuples"):
        lens = [len(x) for x in ds_orig.id_tuples]
        if len(lens) >= 2 and lens[1] > 0:
            start = lens[0]
            end = lens[0] + lens[1]
            for i in range(start, end):
                item = ds_orig[i]
                roles = [t["role"] for t in item["read_prompt"]]
                assert roles == expected_roles, f"Original 'stimulus_completion' roles mismatch at idx {i}: {roles}"
        else:
            print("[warn] Original dataset has no 'stimulus_completion' items to verify.")
    else:
        print("[warn] Original dataset does not expose id_tuples; skipping shape assert.")

    # Preview: first two items from each source in standalone dataset
    print("\nPreview (2 per source, standalone view):")
    standalone.preview_dataset(ds_standalone, per_source=2)


if __name__ == "__main__":
    main()
