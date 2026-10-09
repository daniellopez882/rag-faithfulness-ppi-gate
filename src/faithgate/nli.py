"""Binary NLI scoring with an ONNX model and a Hugging Face fast tokenizer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class EntailmentModel(Protocol):
    """Anything that returns P(premise entails hypothesis) for a batch of pairs."""

    name: str

    def entail_probs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


class OnnxNLI:
    """xtremedistil-style binary NLI (entailment vs not_entailment) on onnxruntime.

    Pairs are tokenised once, sorted by length and batched so that padding stays small; the
    premise is truncated first so the hypothesis is never cut.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        max_length: int = 512,
        batch_size: int = 16,
        threads: int | None = None,
        name: str = "xtremedistil-l6-h256-zeroshot-v1.1-all-33",
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = name
        self.batch_size = batch_size
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        label2id = {k.lower(): int(v) for k, v in config["label2id"].items()}
        if "entailment" not in label2id:
            raise ValueError(f"{model_dir} has no 'entailment' label: {config['label2id']}")
        self._entail_index = label2id["entailment"]
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(max_length=max_length, strategy="only_first")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(model_dir / "onnx" / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

    def entail_probs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float64)
        encodings = self._tokenizer.encode_batch(list(pairs))
        order = np.argsort([len(e.ids) for e in encodings], kind="stable")
        probs = np.empty(len(pairs), dtype=np.float64)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = [encodings[i] for i in idx]
            width = max(len(e.ids) for e in batch)
            ids = np.zeros((len(batch), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            types = np.zeros_like(ids)
            for row, enc in enumerate(batch):
                n = len(enc.ids)
                ids[row, :n] = enc.ids
                mask[row, :n] = 1
                types[row, :n] = enc.type_ids
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = types
            logits = self._session.run(None, feeds)[0].astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            soft = np.exp(logits)
            soft /= soft.sum(axis=1, keepdims=True)
            probs[idx] = soft[:, self._entail_index]
        return probs


def load_default_nli(
    *, max_length: int = 512, batch_size: int = 16, threads: int | None = None
) -> OnnxNLI:
    """The pinned model from the Hugging Face cache; raises with a hint when it was not fetched."""
    from faithgate.fetch import locate_nli_model

    folder = locate_nli_model()
    if folder is None:
        raise FileNotFoundError(
            "NLI model not in the Hugging Face cache; run `faithgate fetch model`"
        )
    return OnnxNLI(folder, max_length=max_length, batch_size=batch_size, threads=threads)
