import json
import regex as re
import heapq
import collections
from pathlib import Path
from datasets import load_dataset

# GPT-4's pre-tokenization pattern (from tiktoken's cl100k_base)
# Splits text into chunks before BPE runs, so merges never cross these boundaries
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|"""
    r"""\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

class Tokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self.pattern = GPT4_SPLIT_PATTERN
        self._compiled_pattern = re.compile(self.pattern)
        # Cache: maps a pre-tokenized chunk of bytes -> list of token ids
        # Goal of cahce is to make repeat words faster to encode
        self._encode_cache: dict[bytes, list[int]] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        chunks = re.findall(self._compiled_pattern, text)
        chunks_ids: list[list[int]] = [list(c.encode("utf-8")) for c in chunks if c]

        pair_counts: collections.Counter = collections.Counter()
        pair_to_chunks: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
        for ci, ids in enumerate(chunks_ids):
            for a, b in zip(ids, ids[1:]):
                pair_counts[(a, b)] += 1
                pair_to_chunks[(a, b)].add(ci)

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 1:
                break

            new_id = 256 + i
            merges[best_pair] = new_id
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

            affected = list(pair_to_chunks[best_pair])
            a, b = best_pair
            for ci in affected:
                ids = chunks_ids[ci]
                new_ids: list[int] = []
                j = 0
                while j < len(ids):
                    if j < len(ids) - 1 and ids[j] == a and ids[j+1] == b:
                        if new_ids:
                            left = new_ids[-1]
                            pair_counts[(left, a)] -= 1
                            if pair_counts[(left, a)] <= 0:
                                del pair_counts[(left, a)]
                            pair_to_chunks[(left, a)].discard(ci)
                            pair_counts[(left, new_id)] += 1
                            pair_to_chunks[(left, new_id)].add(ci)
                        if j + 2 < len(ids):
                            right = ids[j+2]
                            pair_counts[(b, right)] -= 1
                            if pair_counts[(b, right)] <= 0:
                                del pair_counts[(b, right)]
                            pair_to_chunks[(b, right)].discard(ci)
                            pair_counts[(new_id, right)] += 1
                            pair_to_chunks[(new_id, right)].add(ci)
                        new_ids.append(new_id)
                        j += 2
                    else:
                        new_ids.append(ids[j])
                        j += 1
                chunks_ids[ci] = new_ids

            del pair_counts[best_pair]
            del pair_to_chunks[best_pair]

            if verbose and (i < 10 or (i + 1) % 100 == 0):
                print(f"merge {i+1}/{num_merges}: {best_pair} -> {new_id} "
                      f"({vocab[new_id]!r}) had {best_count} occurrences")

        self.merges = merges
        self.vocab = vocab
        self._encode_cache.clear()

    @staticmethod
    def _merge_chunk(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        self.special_tokens = special_tokens
        for tok, idx in special_tokens.items():
            self.vocab[idx] = tok.encode("utf-8")

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        if allow_special and self.special_tokens:
            return self._encode_with_specials(text)
        return self._encode_ordinary(text)

    def _encode_ordinary(self, text: str) -> list[int]:
        chunks = re.findall(self._compiled_pattern, text)
        ids: list[int] = []
        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            ids.extend(self._encode_chunk(chunk_bytes))
        return ids

    def _encode_with_specials(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        parts = re.split(special_pattern, text)
        ids: list[int] = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids
    
    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        if chunk_bytes in self._encode_cache:
            return self._encode_cache[chunk_bytes]
        ids = self._bpe(list(chunk_bytes))
        self._encode_cache[chunk_bytes] = ids
        return ids
    
    def _bpe(self, ids: list[int]) -> list[int]:
        if len(ids) < 2:
            return ids
        
        # Doubly-linked list over positions. prev[i] and next[i] point to neighbors
        n = len(ids)
        tokens = list(ids)
        prev = list(range(-1, n-1))
        nxt = list(range(1, n+1))
        nxt[-1] = -1
        alive = [True] * n

        # Heap entries: (merge_rank, position_left). Position points to the left half of a pair
        heap: list[tuple[int, int]] = []
        for i in range(n-1):
            pair = (tokens[i], tokens[i+1])
            if pair in self.merges:
                heapq.heappush(heap, (self.merges[pair], i))

        while heap:
            rank, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            pair = (tokens[i], tokens[j])
            if pair not in self.merges or self.merges[pair] != rank:
                continue

            new_id = self.merges[pair]
            tokens[i] = new_id
            alive[j] = False
            new_next = nxt[j]
            nxt[i] = new_next
            if new_next != -1:
                prev[new_next] = i

            left = prev[i]
            if left != -1:
                left_pair = (tokens[left], tokens[i])
                if left_pair in self.merges:
                    heapq.heappush(heap, (self.merges[left_pair], left))
            right = nxt[i]
            if right != -1:
                right_pair = (tokens[i], tokens[right])
                if right_pair in self.merges:
                    heapq.heappush(heap, (self.merges[right_pair], i))

        # Walk the linked list to collect surviving tokens in order
        result = []
        i = 0
        while i != -1 and i < n:
            if alive[i]:
                result.append(tokens[i])
            i = nxt[i]
        return result

    def decode(self, ids: list[int]) -> str:
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            else:
                raise ValueError(f"unknown token id: {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "pattern": self.pattern,
            "merges": {f"{a},{b}": new_id for (a, b), new_id in self.merges.items()},
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "special_tokens": self.special_tokens
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        path = Path(path)
        data = json.loads(path.read_text())
        tok = cls()
        tok.pattern = data["pattern"]
        tok._compiled_pattern = re.compile(tok.pattern)
        tok.merges = {
            (lambda parts: (int(parts[0]), int(parts[1])))(k.split(",")): v
            for k, v in data["merges"].items()
        }
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.special_tokens = data["special_tokens"]
        return tok

    def train_from_dataset(
        self,
        dataset_name: str,
        vocab_size: int,
        subset: str | None = None,
        split: str = "train",
        text_field: str = "text",
        sample_size: int | None = None,
        max_chars: int | None = None,
        streaming: bool = False,
        verbose: bool = False,
    ) -> None:
        """Train the tokenizer on a HuggingFace dataset.

        Args:
            dataset_name: HF dataset id (e.g. "roneneldan/TinyStories").
            vocab_size: target vocabulary size.
            subset: optional config name for the dataset (e.g. "sample-10BT").
            split: dataset split to use.
            text_field: name of the text column in the dataset.
            sample_size: max number of documents to use. None = all (non-streaming only).
            max_chars: stop collecting text after this many characters. Useful
                with streaming. None = no limit.
            streaming: stream the dataset instead of downloading fully.
            verbose: forwarded to train().
        """
        print(f"loading {dataset_name}" + (f" (subset: {subset})" if subset else ""))
        ds = load_dataset(dataset_name, name=subset, split=split, streaming=streaming)

        chunks: list[str] = []
        total_chars = 0
        n_docs = 0

        if streaming:
            iterator = ds
        else:
            n_available = len(ds)
            limit = min(sample_size, n_available) if sample_size else n_available
            iterator = (ds[i] for i in range(limit))

        for doc in iterator:
            text = doc[text_field]
            chunks.append(text)
            total_chars += len(text)
            n_docs += 1
            if sample_size is not None and n_docs >= sample_size:
                break
            if max_chars is not None and total_chars >= max_chars:
                break

        text = "\n".join(chunks)
        print(f"training tokenizer on {total_chars/1e6:.1f}MB of text "
              f"from {n_docs:,} documents")

        self.train(text, vocab_size=vocab_size, verbose=verbose)
