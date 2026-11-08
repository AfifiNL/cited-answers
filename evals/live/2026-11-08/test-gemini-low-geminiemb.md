# Eval report

Generated 2026-11-08T18:13:57Z

Config: model=gemini:gemini-3.6-flash:thinking-low, embedder=gemini:gemini-embedding-001:768, store=memory, corpus_version=cbd4f9deaa0bc9cd

| scope | items | faithfulness | citation_precision | citation_hit_rate | refusal_accuracy | unsafe_answer_rate | false_refusal_rate | retrieval_recall |
|---|---|---|---|---|---|---|---|---|
| all | 43 | 1.000 | 0.953 | 0.969 | 0.977 | 0.000 | 0.030 | 1.000 |
| test | 43 | 1.000 | 0.953 | 0.969 | 0.977 | 0.000 | 0.030 | 1.000 |
| ai_act | 17 | 1.000 | 0.912 | 0.941 | 1.000 | 0.000 | 0.000 | 1.000 |
| gdpr | 16 | 1.000 | 1.000 | 1.000 | 0.938 | 0.000 | 0.062 | 1.000 |
| out_of_scope | 10 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Gate

- PASS
- note: baseline was produced with another model/embedder/store: only the floors apply

## Items that were not fully right

| id | split | expected | got | cited | gold | issue |
|---|---|---|---|---|---|---|
| g22 | test | answer | refused (model_declined) |  | gdpr:art4 | wrong refusal decision |
| a12 | test | answer | answered | ai_act:art15, ai_act:art13 | ai_act:art15 | imprecise citation |
| a18 | test | answer | answered | ai_act:art3 | ai_act:art57, ai_act:art58 | imprecise citation |
