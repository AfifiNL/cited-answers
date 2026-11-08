# Eval report

Generated 2026-11-08T18:15:50Z

Config: model=gemini:gemini-3.6-flash:thinking-low, embedder=gemini:gemini-embedding-001:768, store=pgvector, corpus_version=cbd4f9deaa0bc9cd

| scope | items | faithfulness | citation_precision | citation_hit_rate | refusal_accuracy | unsafe_answer_rate | false_refusal_rate | retrieval_recall |
|---|---|---|---|---|---|---|---|---|
| all | 87 | 1.000 | 0.976 | 1.000 | 0.954 | 0.000 | 0.060 | 1.000 |
| dev | 44 | 1.000 | 0.968 | 1.000 | 0.932 | 0.000 | 0.088 | 1.000 |
| test | 43 | 1.000 | 0.984 | 1.000 | 0.977 | 0.000 | 0.030 | 1.000 |
| ai_act | 35 | 1.000 | 0.956 | 1.000 | 0.971 | 0.000 | 0.029 | 1.000 |
| gdpr | 32 | 1.000 | 1.000 | 1.000 | 0.906 | 0.000 | 0.094 | 1.000 |
| out_of_scope | 20 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Gate

- PASS
- note: baseline was produced with another model/embedder/store: only the floors apply

## Items that were not fully right

| id | split | expected | got | cited | gold | issue |
|---|---|---|---|---|---|---|
| g07 | dev | answer | refused (model_declined) |  | gdpr:art83 | wrong refusal decision |
| g22 | test | answer | refused (model_declined) |  | gdpr:art4 | wrong refusal decision |
| g27 | dev | answer | refused (unverified) |  | gdpr:art47 | wrong refusal decision |
| a15 | dev | answer | answered | ai_act:art99, ai_act:art100 | ai_act:art99 | imprecise citation |
| a17 | dev | answer | refused (model_declined) |  | ai_act:art3 | wrong refusal decision |
| a18 | test | answer | answered | ai_act:art3, ai_act:art57 | ai_act:art57, ai_act:art58 | imprecise citation |
| a19 | dev | answer | answered | ai_act:art73, ai_act:art55 | ai_act:art73 | imprecise citation |
