# Eval report

Generated 2026-11-08T18:12:22Z

Config: model=gemini:gemini-3.6-flash:thinking-low, embedder=gemini:gemini-embedding-001:768, store=memory, corpus_version=cbd4f9deaa0bc9cd

| scope | items | faithfulness | citation_precision | citation_hit_rate | refusal_accuracy | unsafe_answer_rate | false_refusal_rate | retrieval_recall |
|---|---|---|---|---|---|---|---|---|
| all | 44 | 1.000 | 0.984 | 1.000 | 0.932 | 0.000 | 0.088 | 1.000 |
| dev | 44 | 1.000 | 0.984 | 1.000 | 0.932 | 0.000 | 0.088 | 1.000 |
| ai_act | 18 | 1.000 | 0.971 | 1.000 | 0.944 | 0.000 | 0.056 | 1.000 |
| gdpr | 16 | 1.000 | 1.000 | 1.000 | 0.875 | 0.000 | 0.125 | 1.000 |
| out_of_scope | 10 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Gate

- PASS
- note: baseline was produced with another model/embedder/store: only the floors apply

## Items that were not fully right

| id | split | expected | got | cited | gold | issue |
|---|---|---|---|---|---|---|
| g07 | dev | answer | refused (model_declined) |  | gdpr:art83 | wrong refusal decision |
| g27 | dev | answer | refused (unverified) |  | gdpr:art47 | wrong refusal decision |
| a15 | dev | answer | answered | ai_act:art99, ai_act:art100 | ai_act:art99 | imprecise citation |
| a17 | dev | answer | refused (model_declined) |  | ai_act:art3 | wrong refusal decision |
