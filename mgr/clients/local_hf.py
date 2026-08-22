"""Local HuggingFace generation client — on-GPU, no external API.

Where sending text to a hosted API is not permitted (e.g. MIMIC under the
PhysioNet DUA), generation must run on a model you control. This client runs a
small instruct model on the local GPU via ``transformers`` and exposes the same
``complete_text`` / ``complete_text_logprobs`` contract as
:class:`mgr.clients.vllm.VLLMClient`, so :class:`~mgr.generate.executor.RAGExecutor`
and Gate-A feature capture work unchanged.

``torch`` / ``transformers`` are imported lazily inside ``__post_init__``, so
importing this module (and the rest of ``mgr``) never requires them — they are
only needed once a client is actually instantiated on the GPU box.

NOTE: this path is GPU/model-dependent and cannot be exercised in the pure-CPU
test env; smoke-test it on the target (Kaggle) before a full run — one item with
`complete_text_logprobs` and check `confidence` in (0,1), `entropy` > 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LocalHFClient:
    """A ``transformers`` causal-LM served locally.

    ``model_id`` is a HF hub id or local path (e.g. ``meta-llama/Llama-3.1-8B-Instruct``).
    Greedy by default (``temperature=0``); pass ``temperature``/``seed`` per call
    for the self-consistency samples.
    """

    model_id: str
    max_new_tokens: int = 8
    temperature: float = 0.0
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    _tok: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        dtype = "auto" if self.torch_dtype == "auto" else getattr(torch, self.torch_dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map=self.device_map, torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()
        # so the executor records a truthful gen_model (getattr(client, "model", ...))
        self.model = self.model_id

    # -- prompt -----------------------------------------------------------------
    def _encode(self, messages: list[dict[str, str]]):
        try:
            text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # model without a chat template -> plain concatenation
            text = "".join(f"{m.get('role','user')}: {m.get('content','')}\n" for m in messages) + "assistant:"
        return self._tok(text, return_tensors="pt").to(self._model.device)

    def _usage(self, n_in: int, n_out: int) -> dict[str, int]:
        return {"in": int(n_in), "out": int(n_out)}

    # -- generation -------------------------------------------------------------
    def complete_text(self, model: str, messages: list[dict[str, str]], **params: Any) -> tuple[str, dict[str, int]]:
        torch = self._torch
        seed = params.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
        temp = float(params.get("temperature", self.temperature) or 0.0)
        max_new = int(params.get("max_tokens", self.max_new_tokens))
        enc = self._encode(messages)
        gen_kw: dict[str, Any] = dict(max_new_tokens=max_new, do_sample=temp > 0.0,
                                      pad_token_id=self._tok.eos_token_id)
        if temp > 0.0:
            gen_kw["temperature"] = temp
        with torch.no_grad():
            out = self._model.generate(**enc, **gen_kw)
        n_in = enc["input_ids"].shape[1]
        new = out[0][n_in:]
        text = self._tok.decode(new, skip_special_tokens=True)
        return text, self._usage(n_in, new.shape[0])

    def complete_text_logprobs(
        self, model: str, messages: list[dict[str, str]], *, top_logprobs: int = 10, **params: Any
    ) -> tuple[str, dict[str, int], list[dict[str, Any]]]:
        """Greedy answer plus the first generated token's top-k logprobs, packaged
        in the OpenAI ``logprobs.content`` shape so ``mgr.gate.features`` parses it
        exactly as it does the API path."""
        torch = self._torch
        seed = params.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
        temp = float(params.get("temperature", self.temperature) or 0.0)
        max_new = int(params.get("max_tokens", self.max_new_tokens))
        enc = self._encode(messages)
        gen_kw: dict[str, Any] = dict(max_new_tokens=max_new, do_sample=temp > 0.0,
                                      pad_token_id=self._tok.eos_token_id,
                                      output_scores=True, return_dict_in_generate=True)
        if temp > 0.0:
            gen_kw["temperature"] = temp
        with torch.no_grad():
            out = self._model.generate(**enc, **gen_kw)
        n_in = enc["input_ids"].shape[1]
        new = out.sequences[0][n_in:]
        text = self._tok.decode(new, skip_special_tokens=True)

        content: list[dict[str, Any]] = []
        if len(new) and out.scores:
            logp = torch.log_softmax(out.scores[0][0], dim=-1)  # first new-token logprobs [vocab]
            k = min(int(top_logprobs), logp.shape[-1])
            vals, idx = torch.topk(logp, k)
            top = [{"token": self._tok.decode([int(i)]), "logprob": float(v)}
                   for v, i in zip(vals.tolist(), idx.tolist())]
            first_id = int(new[0])
            content = [{"token": self._tok.decode([first_id]),
                        "logprob": float(logp[first_id]),
                        "top_logprobs": top}]
        return text, self._usage(n_in, new.shape[0]), content
