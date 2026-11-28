"""Pinned upstream config.json snapshots and runtime Decode dimensions."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"
DEFAULT_INITIAL_KV = {"llama3_70b": 8190, "deepseek_r1": 131072}
SOURCES = {
    "llama3_70b": {
        "repository": "meta-llama/Meta-Llama-3-70B",
        "revision": "c82494877ce7f6d7d317c56ec081328e382c72fe",
        "config_git_blob": "2f26ee5614fe1336b244da1655ba55436f3f1499",
        "mirror": "NousResearch/Meta-Llama-3-70B",
        "mirror_revision": "fdbf133d993d2b8157e518bfc8bc4570b2d31c45",
    },
    "deepseek_r1": {
        "repository": "deepseek-ai/DeepSeek-R1",
        "revision": "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad",
        "config_git_blob": "c482e777cbacba9cb292d0fceba48ef52956583e",
    },
}


def load_model(name):
    if name not in SOURCES:
        raise ValueError(f"Unknown model: {name}")
    # Git checkouts on Windows can use CRLF for the same upstream blob.
    content = (CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8").encode()
    header = f"blob {len(content)}\0".encode()
    if hashlib.sha1(header + content).hexdigest() != SOURCES[name]["config_git_blob"]:
        raise ValueError(f"{name} config differs from the pinned source")
    return json.loads(content)


@dataclass(frozen=True)
class DecodeRequest:
    batch: int
    sequence_length: int
    decode_steps: int

    @classmethod
    def for_model(cls, model, batch, sequence_length, decode_steps):
        initial = DEFAULT_INITIAL_KV[model] if sequence_length is None else sequence_length
        return cls(batch, initial, decode_steps)

    def __post_init__(self):
        for name in ("batch", "sequence_length", "decode_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def effective_lengths(self, config):
        end = self.sequence_length + self.decode_steps
        if end > config["max_position_embeddings"]:
            raise ValueError("Final Decode length exceeds the model context limit")
        return range(self.sequence_length + 1, end + 1)
