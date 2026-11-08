# Eval report

Generated 2026-11-08T18:10:26Z

Config: model=gemini:gemini-3.6-flash, embedder=hash-768, store=memory, corpus_version=cbd4f9deaa0bc9cd

| scope | items | faithfulness | citation_precision | citation_hit_rate | refusal_accuracy | unsafe_answer_rate | false_refusal_rate | retrieval_recall |
|---|---|---|---|---|---|---|---|---|
| all | 44 | 1.000 | 0.928 | 1.000 | 0.909 | 0.000 | 0.118 | 0.941 |
| dev | 44 | 1.000 | 0.928 | 1.000 | 0.909 | 0.000 | 0.118 | 0.941 |
| ai_act | 18 | 1.000 | 0.941 | 1.000 | 0.944 | 0.000 | 0.056 | 1.000 |
| gdpr | 16 | 1.000 | 0.910 | 1.000 | 0.812 | 0.000 | 0.188 | 0.875 |
| out_of_scope | 10 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Gate

- PASS
- note: baseline was produced with another model/embedder/store: only the floors apply

## Items that were not fully right

| id | split | expected | got | cited | gold | issue |
|---|---|---|---|---|---|---|
| g03 | dev | answer | refused (model_declined) |  | gdpr:art6 | wrong refusal decision |
| g07 | dev | answer | refused (model_declined) |  | gdpr:art83 | wrong refusal decision |
| g23 | dev | answer | answered | gdpr:art82, gdpr:art80 | gdpr:art82 | imprecise citation |
| g25 | dev | answer | refused (model_declined) |  | gdpr:art3 | wrong refusal decision |
| g27 | dev | answer | answered | gdpr:art70, gdpr:art57, gdpr:art47 | gdpr:art47 | imprecise citation |
| a13 | dev | answer | answered | ai_act:art53, ai_act:art54 | ai_act:art53, ai_act:art55 | imprecise citation |
| a15 | dev | answer | answered | ai_act:art99, ai_act:art100 | ai_act:art99 | imprecise citation |
| a17 | dev | answer | refused (model_declined) |  | ai_act:art3 | wrong refusal decision |
