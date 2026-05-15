import json
import regex as re
import heapq
import collections
from pathlib import Path

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

        # Pre-tokenize: split text into chunks, then convert each to a list of byte ids
        chunks = re.findall(self._compiled_pattern, text)
        ids: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in chunks]

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            # Count pair frequencies across all chunks
            counts: collections.Counter = collections.Counter()
            for chunk_ids in ids:
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    counts[pair] += 1

            if not counts:
                break

            best_pair = counts.most_common(1)[0][0]
            new_id = 256 + i

            ids = [self._merge_chunk(chunk, best_pair, new_id) for chunk in ids]

            merges[best_pair] = new_id
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {best_pair} -> {new_id} "
                      f"({vocab[new_id]!r}) had {counts[best_pair]} occurrences")
            
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
        next = list(range(1, n+1))
        next[-1] = -1
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
            j = next[i]
            if j == -1 or not alive[j]:
                continue
            pair = (tokens[i], tokens[j])
            if pair not in self.merges or self.merges[pair] != rank:
                continue

            new_id = self.merges[pair]
            tokens[i] = new_id
            alive[j] = False
            new_next = next[j]
            next[i] = new_next
            if new_next != -1:
                prev[new_next] = i

            left = prev[i]
            if left != -1:
                left_pair = (tokens[left], tokens[i])
                if left_pair in self.merges:
                    heapq.heappush(heap, (self.merges[left_pair], left))
            right = next[i]
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
            i = next[i]
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
