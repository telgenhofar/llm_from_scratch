import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm
from tokenizer import Tokenizer


def tokenize_example(example: dict, tokenizer: Tokenizer, text_field: str, eot_token: int) -> dict:
    ids = tokenizer.encode(example[text_field])
    ids.append(eot_token)
    return {"ids": ids, "len": len(ids)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="YAML data config file")
    p.add_argument("--tokenizer", type=str, default="tokenizer.json",
                   help="Path to trained tokenizer")
    p.add_argument("--num_proc", type=int, default=8,
                   help="Worker processes for tokenization")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.load(args.tokenizer)
    # The special token used as a document separator
    eot_token = tokenizer.special_tokens.get("<|endoftext|>")
    if eot_token is None:
        raise ValueError("Tokenizer must have '<|endoftext|>' registered as a special token")

    # Stream dataset from HuggingFace
    print(f"loading {cfg['dataset']} (config: {cfg.get('subset')})")
    ds = load_dataset(
        cfg["dataset"],
        name=cfg.get("subset"),
        split=cfg.get("split", "train"),
        streaming=cfg.get("streaming", False),
        num_proc=args.num_proc if not cfg.get("streaming") else None,
    )

    val_fraction = cfg.get("val_fraction", 0.005)

    text_field = cfg.get("text_field", "text")

    if cfg.get("streaming"):
        write_streaming(ds, tokenizer, text_field, eot_token, out_dir, cfg, val_fraction)
    else:
        write_mapped(ds, tokenizer, text_field, eot_token, out_dir, args.num_proc, val_fraction)


def write_mapped(ds, tokenizer, text_field, eot_token, out_dir, num_proc, val_fraction):
    print(f"dataset size: {len(ds):,} documents")

    split = ds.train_test_split(test_size=val_fraction, seed=42, shuffle=True)

    for name, subset in [("train", split["train"]), ("val", split["test"])]:
        print(f"\ntokenizing {name} split ({len(subset):,} docs)...")
        tokenized = subset.map(
            tokenize_example,
            fn_kwargs={"tokenizer": tokenizer, "text_field": text_field, "eot_token": eot_token},
            remove_columns=subset.column_names,
            desc=f"tokenizing {name}",
            num_proc=num_proc,
        )

        total_tokens = sum(tokenized["len"])
        out_path = out_dir / f"{name}.bin"
        print(f"writing {total_tokens:,} tokens to {out_path}")

        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
        idx = 0
        n_shards = 1024
        for shard_idx in tqdm(range(n_shards), desc=f"writing {name}"):
            shard = tokenized.shard(num_shards=n_shards, index=shard_idx,
                                    contiguous=True).with_format("numpy")
            shard_arr = np.concatenate(shard["ids"])
            arr[idx:idx + len(shard_arr)] = shard_arr
            idx += len(shard_arr)
        arr.flush()

        write_metadata(out_dir / f"{name}.meta.json", total_tokens, tokenizer)


def write_streaming(ds, tokenizer, text_field, eot_token, out_dir, cfg, val_fraction):
    max_tokens = cfg.get("max_tokens")
    if max_tokens is None:
        raise ValueError("Streaming configs must specify max_tokens")

    val_tokens = int(max_tokens * val_fraction)
    train_tokens = max_tokens - val_tokens
    print(f"target: {train_tokens:,} train + {val_tokens:,} val tokens")

    train_arr = np.memmap(out_dir / "train.bin", dtype=np.uint16, mode="w+", shape=(train_tokens,))
    val_arr = np.memmap(out_dir / "val.bin", dtype=np.uint16, mode="w+", shape=(val_tokens,))

    train_idx, val_idx = 0, 0
    pbar = tqdm(total=max_tokens, desc="tokenizing", unit="tok", unit_scale=True)

    for doc in ds:
        ids = tokenizer.encode(doc[text_field])
        ids.append(eot_token)
        n = len(ids)
        if val_idx < val_tokens:
            take = min(n, val_tokens - val_idx)
            val_arr[val_idx:val_idx + take] = ids[:take]
            val_idx += take
            ids = ids[take:]
            n = len(ids)
        if n == 0:
            continue
        if train_idx + n > train_tokens:
            n = train_tokens - train_idx
            train_arr[train_idx:train_idx + n] = ids[:n]
            train_idx += n
            pbar.update(n + take if val_idx else n)
            break
        train_arr[train_idx:train_idx + n] = ids
        train_idx += n
        pbar.update(n + (take if val_idx == val_tokens and train_idx == n else 0))
        if train_idx >= train_tokens:
            break
    pbar.close()

    train_arr.flush()
    val_arr.flush()
    write_metadata(out_dir / "train.meta.json", train_idx, tokenizer)
    write_metadata(out_dir / "val.meta.json", val_idx, tokenizer)


def write_metadata(path: Path, n_tokens: int, tokenizer: Tokenizer) -> None:
    meta = {
        "n_tokens": n_tokens,
        "vocab_size": len(tokenizer.vocab),
        "dtype": "uint16",
    }
    path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()