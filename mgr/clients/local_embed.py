"""Local embedder — dense vectors computed in-process, no external API.

Mirrors :class:`mgr.clients.nim_adapters.NimEmbedder` and satisfies the same
``Embedder = Callable[[list[str]], np.ndarray]`` contract that
:mod:`mgr.retrieval.dense` consumes, so it drops into ``DenseIndex.from_corpus``
and ``DenseRetriever`` with no other change.

Two reasons this exists, both load-bearing:

* **Governance.** Under the PhysioNet DUA no chart text may leave the machine, so
  a hosted embeddings endpoint is not an option — the same constraint that
  produced :class:`mgr.clients.local_hf.LocalHFClient`.
* **Throughput.** The NIM free tier is ~40 requests/minute. Embedding a 126k-chunk
  corpus through it takes weeks; locally it takes minutes. The 600-chunk corpus
  cap the POC ran under was a symptom of that limit, not a design choice, and it
  bounds how much retrieval quality that run could ever have shown.

MedCPT is **asymmetric**: queries and passages have separate encoders. Using one
for both silently degrades retrieval without failing, so the two checkpoints are
named constants here and :func:`medcpt_pair` hands back the matched pair.

``torch``/``transformers`` are imported lazily in :meth:`load`, so importing this
module never requires them. Injecting ``encode_batch`` exercises the batching and
normalization logic without a model — the pattern
:class:`mgr.rerank.cross_encoder.HFCrossEncoderReranker` already uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

#: MedCPT's two halves. Passages go through the article encoder, queries through
#: the query encoder — see :func:`medcpt_pair`.
MEDCPT_ARTICLE = "ncbi/MedCPT-Article-Encoder"
MEDCPT_QUERY = "ncbi/MedCPT-Query-Encoder"

EncodeBatch = Callable[[list[str]], np.ndarray]


@dataclass
class LocalEmbedder:
    """A HuggingFace encoder served locally, callable as an ``Embedder``.

    ``pooling`` is ``"cls"`` for MedCPT (and most BERT-style retrieval encoders);
    ``"mean"`` covers mean-pooled checkpoints. Vectors are L2-normalized on the
    way out so cosine similarity is a plain dot product — ``DenseIndex``
    normalizes again, which is harmless and keeps either path correct.
    """

    model_id: str = MEDCPT_ARTICLE
    max_length: int = 512
    batch_size: int = 64
    device: str | None = None
    pooling: str = "cls"
    normalize: bool = True
    encode_batch: EncodeBatch | None = None
    _loaded: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.pooling not in {"cls", "mean"}:
            raise ValueError(f"pooling must be 'cls' or 'mean', got {self.pooling!r}")

    def load(self) -> "LocalEmbedder":
        """Materialise the model. Idempotent; a no-op if ``encode_batch`` is set."""
        if self.encode_batch is not None or self._loaded:
            return self
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModel.from_pretrained(self.model_id).eval()
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        def _encode(texts: list[str]) -> np.ndarray:
            enc = tok(
                texts, truncation=True, padding=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**enc).last_hidden_state
            if self.pooling == "cls":
                vecs = out[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).type_as(out)
                vecs = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return vecs.detach().cpu().numpy()

        self.encode_batch = _encode
        self._loaded = True
        return self

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` in batches -> (n, dim) float array."""
        items = [str(t) for t in texts]
        if not items:
            return np.zeros((0, 0), dtype=float)
        if self.encode_batch is None:
            self.load()
        assert self.encode_batch is not None
        out = [
            np.asarray(self.encode_batch(items[i : i + self.batch_size]), dtype=float)
            for i in range(0, len(items), self.batch_size)
        ]
        matrix = np.vstack(out)
        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix


def medcpt_pair(**kw) -> tuple[LocalEmbedder, LocalEmbedder]:
    """``(passage_embedder, query_embedder)`` — the matched MedCPT halves.

    Build the index with the first and retrieve with the second. Keeping them
    together is the point: a single embedder used for both is a silent accuracy
    loss, not an error.
    """
    return (
        LocalEmbedder(model_id=MEDCPT_ARTICLE, **kw),
        LocalEmbedder(model_id=MEDCPT_QUERY, **kw),
    )
