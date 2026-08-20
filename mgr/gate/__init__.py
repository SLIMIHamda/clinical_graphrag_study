"""Gate-A: pre-retrieval selective-retrieval policy.

Gate A decides, *before* retrieval, whether a question needs the corpus at all,
using only signals from the base model's No-RAG pass. See docs/gate_a_phase_a.md
for the reward framing, the instrumentation contract, and the Phase-A plan.

- ``mgr.gate.features``          -- turn raw generation signals into the feature vector x_q
- ``mgr.analysis.gate_signal``  -- Phase-A: does x_q predict the retrieval reward?
"""
