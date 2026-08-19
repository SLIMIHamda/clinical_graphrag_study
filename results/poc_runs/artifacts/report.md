# Results

## Run

| key | value |
| --- | --- |
| mode | NIM |
| reranker | local:ncbi/MedCPT-Cross-Encoder |
| generation_actual | nim:meta/llama-3.1-8b-instruct |
| manifest_backbone | Llama-70B |
| gen_model_recorded | ['meta/llama-3.1-8b-instruct'] |
| provenance_complete | True |
| model_substituted_for_backbone | True |
| data_source | real |
| benchmark | MedQA-US |
| umls_source | scispacy |
| conditions | ['No-RAG', 'BM25', 'Dense-MedCPT', 'Graph-only', 'Hybrid-CARRF', 'Hybrid-CARRF-CARe'] |
| n_items | 256 |
| gates | H2=True, G3=True, P3=True |
| n_runs | 18 |
| n_arms_failed | 0 |
| run_dir | /kaggle/working/poc_runs |

## Arms

| condition | seed | n_items | accuracy | coverage | item_errors | ctx_precision(RAGAS) | tokens | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 42 | 256 | 0.555 | 1.0 | 0 | 0.111 | 551781 | Done |
| BM25 | 7 | 256 | 0.559 | 1.0 | 0 | 0.111 | 551777 | Done |
| BM25 | 123 | 256 | 0.566 | 1.0 | 0 | 0.111 | 551776 | Done |
| Dense-MedCPT | 123 | 256 | 0.551 | 1.0 | 0 | 0.625 | 567514 | Done |
| Dense-MedCPT | 42 | 256 | 0.555 | 1.0 | 0 | 0.625 | 567748 | Done |
| Dense-MedCPT | 7 | 256 | 0.555 | 1.0 | 0 | 0.625 | 567748 | Done |
| Graph-only | 42 | 256 | 0.535 | 1.0 | 0 | 0.167 | 531241 | Done |
| Graph-only | 7 | 256 | 0.547 | 1.0 | 0 | 0.167 | 531241 | Done |
| Graph-only | 123 | 256 | 0.562 | 1.0 | 0 | 0.167 | 531237 | Done |
| Hybrid-CARRF | 123 | 256 | 0.535 | 1.0 | 0 | 0.417 | 537300 | Done |
| Hybrid-CARRF | 7 | 256 | 0.551 | 1.0 | 0 | 0.417 | 537302 | Done |
| Hybrid-CARRF | 42 | 256 | 0.547 | 1.0 | 0 | 0.417 | 537303 | Done |
| Hybrid-CARRF-CARe | 42 | 256 | 0.543 | 1.0 | 0 | 0.375 | 537150 | Done |
| Hybrid-CARRF-CARe | 123 | 256 | 0.547 | 1.0 | 0 | 0.375 | 537311 | Done |
| Hybrid-CARRF-CARe | 7 | 256 | 0.539 | 1.0 | 0 | 0.375 | 537299 | Done |
| No-RAG | 123 | 256 | 0.594 | 1.0 | 0 | 0.0 | 72678 | Done |
| No-RAG | 7 | 256 | 0.59 | 1.0 | 0 | 0.0 | 72676 | Done |
| No-RAG | 42 | 256 | 0.598 | 1.0 | 0 | 0.0 | 72677 | Done |

## Retrieval rescue (vs No-RAG)

| condition | n_items | arm_acc | base_acc | base_wrong | rescues | rescue_rate | base_right | breaks | break_rate | net_items | net_acc_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 768 | 0.5599 | 0.5938 | 312 | 40 | 0.1282 | 456 | 66 | 0.1447 | -26 | -0.0339 |
| Dense-MedCPT | 768 | 0.5534 | 0.5938 | 312 | 41 | 0.1314 | 456 | 72 | 0.1579 | -31 | -0.0404 |
| Graph-only | 768 | 0.5482 | 0.5938 | 312 | 33 | 0.1058 | 456 | 68 | 0.1491 | -35 | -0.0456 |
| Hybrid-CARRF | 768 | 0.5443 | 0.5938 | 312 | 41 | 0.1314 | 456 | 79 | 0.1732 | -38 | -0.0495 |
| Hybrid-CARRF-CARe | 768 | 0.543 | 0.5938 | 312 | 40 | 0.1282 | 456 | 79 | 0.1732 | -39 | -0.0508 |

## Gate-A retrieval oracle (selective-retrieval ceiling)

| condition | n_items | base_acc | arm_acc | oracle_acc | oracle_gain_vs_norag | oracle_gain_vs_arm |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 768 | 0.5938 | 0.5599 | 0.6458 | 0.0521 | 0.0859 |
| Dense-MedCPT | 768 | 0.5938 | 0.5534 | 0.6471 | 0.0534 | 0.0938 |
| Graph-only | 768 | 0.5938 | 0.5482 | 0.6367 | 0.043 | 0.0885 |
| Hybrid-CARRF | 768 | 0.5938 | 0.5443 | 0.6471 | 0.0534 | 0.1029 |
| Hybrid-CARRF-CARe | 768 | 0.5938 | 0.543 | 0.6458 | 0.0521 | 0.1029 |

## Gates

| key | value |
| --- | --- |
| H2 | True |
| G3 | True |
| P3 | True |

## Graph (G3)

| key | value |
| --- | --- |
| graph_hash | 9e1abb1edf5399d29ffdfe03c4a66c394178dd865e122b4683c2e9067b331280 |
| n_chunks | 792 |
| n_concepts | 4171 |
| n_links | 14564 |
| coverage | {"exact": 0.5721844546398313, "exact+abbrev": 0.5724858511101437, "exact+abbrev+fuzzy": 0.6009175848096179} |
| umls_source | scispacy |
| linked_frac | 0.6009 |
| coverage_floor | 0.2 |
| docs_with_concepts | 779 |
| G3 | True |

## CARe oracle (P3)

| key | value |
| --- | --- |
| n_items | 64 |
| positive_rate | 0.0625 |
| rerank_changed_top_k | 64 |
| acc_with_rerank | 0.5625 |
| acc_without_rerank | 0.5625 |
| P3 | True |

## RAGAS grounding

| key | value |
| --- | --- |
| faithfulness | 0.148 |
| answer_relevance | 0.869 |
| context_precision | 0.417 |
| context_recall | 0.091 |
| n_items | 12 |

## Per-run records

| run_id | condition | backbone | gen_model | seed | status | accuracy | n_item_errors | tokens_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0004 | No-RAG | Llama-70B | meta/llama-3.1-8b-instruct | 42 | Done | 0.5977 | 0 | 72677 |
| R0005 | No-RAG | Llama-70B | meta/llama-3.1-8b-instruct | 123 | Done | 0.5938 | 0 | 72678 |
| R0006 | No-RAG | Llama-70B | meta/llama-3.1-8b-instruct | 7 | Done | 0.5898 | 0 | 72676 |
| R0028 | BM25 | Llama-70B | meta/llama-3.1-8b-instruct | 42 | Done | 0.5547 | 0 | 551781 |
| R0029 | BM25 | Llama-70B | meta/llama-3.1-8b-instruct | 123 | Done | 0.5664 | 0 | 551776 |
| R0030 | BM25 | Llama-70B | meta/llama-3.1-8b-instruct | 7 | Done | 0.5586 | 0 | 551777 |
| R0052 | Dense-MedCPT | Llama-70B | meta/llama-3.1-8b-instruct | 42 | Done | 0.5547 | 0 | 567748 |
| R0053 | Dense-MedCPT | Llama-70B | meta/llama-3.1-8b-instruct | 123 | Done | 0.5508 | 0 | 567514 |
| R0054 | Dense-MedCPT | Llama-70B | meta/llama-3.1-8b-instruct | 7 | Done | 0.5547 | 0 | 567748 |
| R0076 | Graph-only | Llama-70B | meta/llama-3.1-8b-instruct | 42 | Done | 0.5352 | 0 | 531241 |
| R0077 | Graph-only | Llama-70B | meta/llama-3.1-8b-instruct | 123 | Done | 0.5625 | 0 | 531237 |
| R0078 | Graph-only | Llama-70B | meta/llama-3.1-8b-instruct | 7 | Done | 0.5469 | 0 | 531241 |
| R0148 | Hybrid-CARRF | Llama-70B | meta/llama-3.1-8b-instruct | 42 | Done | 0.5469 | 0 | 537303 |
| R0149 | Hybrid-CARRF | Llama-70B | meta/llama-3.1-8b-instruct | 123 | Done | 0.5352 | 0 | 537300 |
| R0150 | Hybrid-CARRF | Llama-70B | meta/llama-3.1-8b-instruct | 7 | Done | 0.5508 | 0 | 537302 |
| R0172 | Hybrid-CARRF-CARe | Llama-70B | meta/llama-3.1-8b-instruct | 42 | Done | 0.543 | 0 | 537150 |
| R0173 | Hybrid-CARRF-CARe | Llama-70B | meta/llama-3.1-8b-instruct | 123 | Done | 0.5469 | 0 | 537311 |
| R0174 | Hybrid-CARRF-CARe | Llama-70B | meta/llama-3.1-8b-instruct | 7 | Done | 0.5391 | 0 | 537299 |

## Figures

**F3 — Retrieval–Generation Decomposition (C1)**

![F3 — Retrieval–Generation Decomposition (C1)](figures/F3_rgd.png)

**F4 — CARe cost–quality frontier (C3)**

![F4 — CARe cost–quality frontier (C3)](figures/F4_pareto.png)

**F5 — UMLS coverage curve**

![F5 — UMLS coverage curve](figures/F5_coverage.png)
