# Eval report

Generated 2026-11-08T18:08:04Z

Config: model=mock-extractive, embedder=gemini:gemini-embedding-001:768, store=memory, corpus_version=cbd4f9deaa0bc9cd

| scope | items | faithfulness | citation_precision | citation_hit_rate | refusal_accuracy | unsafe_answer_rate | false_refusal_rate | retrieval_recall |
|---|---|---|---|---|---|---|---|---|
| all | 44 | 1.000 | 0.882 | 0.882 | 0.977 | 0.100 | 0.000 | 1.000 |
| dev | 44 | 1.000 | 0.882 | 0.882 | 0.977 | 0.100 | 0.000 | 1.000 |
| ai_act | 18 | 1.000 | 0.944 | 0.944 | 1.000 | 0.000 | 0.000 | 1.000 |
| gdpr | 16 | 1.000 | 0.812 | 0.812 | 1.000 | 0.000 | 0.000 | 1.000 |
| out_of_scope | 10 | 1.000 | 0.000 | 0.000 | 0.900 | 0.100 | 0.000 | 0.000 |

## Gate

- PASS
- note: baseline was produced with another model/embedder/store: only the floors apply

## Items that were not fully right

| id | split | expected | got | cited | gold | issue |
|---|---|---|---|---|---|---|
| g03 | dev | answer | answered | gdpr:art28 | gdpr:art6 | imprecise citation |
| g13 | dev | answer | answered | gdpr:art47 | gdpr:art5 | imprecise citation |
| g25 | dev | answer | answered | gdpr:art27 | gdpr:art3 | imprecise citation |
| a15 | dev | answer | answered | ai_act:art100 | ai_act:art99 | imprecise citation |
| u09 | dev | refuse | answered | ai_act:art50 |  | wrong refusal decision |
