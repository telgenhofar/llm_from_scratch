# train_tokenizer.py
import random
from tokenizer import Tokenizer


def verify_correctness():
    """Quick sanity check that train() produces correct, roundtrippable encodings."""
    print("running correctness check...")
    random.seed(0)
    text = " ".join(random.choices(
        ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "quietly"],
        k=500,
    ))

    tok = Tokenizer()
    tok.train(text, vocab_size=320, verbose=False)

    # 1. Merges were actually learned
    assert len(tok.merges) > 0, "no merges learned"
    
    # 2. Vocab grew correctly
    assert len(tok.vocab) == 256 + len(tok.merges), \
        f"vocab size mismatch: {len(tok.vocab)} vs {256 + len(tok.merges)}"
    
    # 3. Encoding then decoding roundtrips
    sample = "the cat sat on the mat"
    decoded = tok.decode(tok.encode(sample))
    assert decoded == sample, f"roundtrip failed: {decoded!r} != {sample!r}"

    # 4. Encoding produces fewer tokens than the raw byte count (compression actually happened)
    raw_byte_count = len(sample.encode("utf-8"))
    encoded_count = len(tok.encode(sample))
    assert encoded_count < raw_byte_count, \
        f"no compression: {encoded_count} tokens for {raw_byte_count} bytes"

    print(f"  passed: {len(tok.merges)} merges, "
          f"{raw_byte_count} bytes -> {encoded_count} tokens")


if __name__ == "__main__":
    verify_correctness()

    print("\ntraining production tokenizer on TinyStories...")
    tok = Tokenizer()
    tok.train_from_dataset(
        "roneneldan/TinyStories",
        vocab_size=4096,
        sample_size=100_000,
        verbose=True,
    )
    tok.register_special_tokens({"<|endoftext|>": 4096})
    tok.save("tokenizer.json")
    print(f"\nsaved tokenizer.json with vocab size {len(tok.vocab)}")