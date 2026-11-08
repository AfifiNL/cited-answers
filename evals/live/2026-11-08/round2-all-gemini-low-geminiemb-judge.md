# Eval report

Generated 2026-11-08T19:48:43Z

Config: model=gemini:gemini-3.6-flash:thinking-low, embedder=gemini:gemini-embedding-001:768, store=memory, scorer=judge:gemini:gemini-3.1-pro-preview, corpus_version=cbd4f9deaa0bc9cd

| scope | items | faithfulness | citation_precision | citation_hit_rate | refusal_accuracy | unsafe_answer_rate | false_refusal_rate | retrieval_recall |
|---|---|---|---|---|---|---|---|---|
| all | 87 | 0.884 | 0.939 | 0.985 | 0.977 | 0.000 | 0.030 | 1.000 |
| dev | 44 | 0.827 | 0.938 | 1.000 | 0.955 | 0.000 | 0.059 | 1.000 |
| test | 43 | 0.946 | 0.939 | 0.970 | 1.000 | 0.000 | 0.000 | 1.000 |
| ai_act | 35 | 0.860 | 0.914 | 0.971 | 1.000 | 0.000 | 0.000 | 1.000 |
| gdpr | 32 | 0.911 | 0.967 | 1.000 | 0.938 | 0.000 | 0.062 | 1.000 |
| out_of_scope | 20 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Gate

- FAIL: all.faithfulness = 0.8842 is below the floor 0.9
- FAIL: dev.faithfulness = 0.8265 is below the floor 0.9
- FAIL: 1 judge call(s) failed, so faithfulness is not trustworthy: AI Act Article 53(1): Gemini call failed: ClientError (HTTP 429)
- note: baseline was produced with another model/embedder/store/scorer: only the floors apply

## Items that were not fully right

| id | split | expected | got | cited | gold | issue |
|---|---|---|---|---|---|---|
| g01 | dev | answer | answered | gdpr:art17 | gdpr:art17 | judge: partially_supported: The provision establishes the right to erasure when the data are no longer necessary for their original purposes, but the claim omits the exceptions listed in paragraph 3 where this right does not apply. |
| g03 | dev | answer | answered | gdpr:art6 | gdpr:art6 | judge: partially_supported: The claim accurately paraphrases the condition for legitimate interests, but it drops an important exception stated at the end of the paragraph: that this lawful basis does not apply to processing carried out by public authorities in the performance of their tasks. |
| g05 | dev | answer | answered | gdpr:art35 | gdpr:art35 | judge: partially_supported: The claim broadly reflects Article 35(10), but omits the condition that the exception only applies to processing carried out pursuant to points (c) or (e) of Article 6(1). |
| g07 | dev | answer | refused (model_declined) |  | gdpr:art83 | wrong refusal decision |
| g09 | dev | answer | answered | gdpr:art13 | gdpr:art13 | judge: partially_supported: The provision supports the claim that the controller must state the legitimate interests pursued, but the claim drops the condition that this specific obligation applies when personal data are collected from the data subject.; judge: partially_supported: The claim states as a general obligation what the provision limits to cases where personal data are collected from the data subject. |
| g13 | dev | answer | answered | gdpr:art5 | gdpr:art5 | judge: partially_supported: The claim accurately states the general rule of storage limitation, but omits the exception in the provision that allows personal data to be stored for longer periods for archiving in the public interest, scientific or historical research, or statistical purposes. |
| g15 | dev | answer | refused (unverified) |  | gdpr:art30 | wrong refusal decision |
| g18 | test | answer | answered | gdpr:art34 | gdpr:art34 | judge: partially_supported: The claim accurately reflects the general rule for communicating a data breach as stated in paragraph 1, but it omits the exceptions listed in paragraph 3 where this communication is not required. |
| g19 | dev | answer | answered | gdpr:art8 | gdpr:art8 | judge: partially_supported: The claim accurately reflects the age requirement for lawful processing of a child's data for information society services, but it omits the crucial condition that this only applies where processing is based on consent (Article 6(1)(a)). |
| g27 | dev | answer | answered | gdpr:art4, gdpr:art47 | gdpr:art47 | imprecise citation |
| g28 | test | answer | answered | gdpr:art39, gdpr:art38 | gdpr:art39 | imprecise citation |
| a02 | test | answer | answered | ai_act:art50 | ai_act:art50 | judge: partially_supported: The claim accurately captures the general rule for providers of AI systems intended to interact directly with natural persons, but it omits the exception for AI systems authorised by law to detect, prevent, investigate, or prosecute criminal offences. |
| a03 | dev | answer | answered | ai_act:art4 | ai_act:art4 | judge: partially_supported: The claim correctly states the obligation to ensure AI literacy, but omits the factors that must be taken into account when doing so, such as technical knowledge, experience, and the context of use. |
| a04 | test | answer | answered | ai_act:art6 | ai_act:art6, ai_act:anxIII | judge: partially_supported: The claim accurately reflects the general rule from Article 6(2), but omits the important exception in Article 6(3) which states that Annex III AI systems are not considered high-risk if they do not pose a significant risk of harm.; judge: partially_supported: The claim accurately states the general rule that an Annex III AI system is not high-risk if it does not pose a significant risk and meets specified conditions. However, it omits the crucial exception that such systems shall always be considered high-risk if they perform profiling of natural persons. |
| a05 | dev | answer | answered | ai_act:art26, ai_act:art25 | ai_act:art26 | imprecise citation |
| a12 | test | answer | answered | ai_act:art15, ai_act:art13 | ai_act:art15 | imprecise citation |
| a13 | dev | answer | answered | ai_act:art53, ai_act:art54 | ai_act:art53, ai_act:art55 | imprecise citation |
| a15 | dev | answer | answered | ai_act:art99, ai_act:art100 | ai_act:art99 | imprecise citation |
| a18 | test | answer | answered | ai_act:art3 | ai_act:art57, ai_act:art58 | imprecise citation |
| a19 | dev | answer | answered | ai_act:art73 | ai_act:art73 | judge: partially_supported: The claim accurately reflects the general reporting deadline set out in Article 73(2), but it omits the important exceptions in paragraphs 3 and 4, which impose shorter deadlines (two or ten days) for specific cases such as widespread infringement or death.; judge: partially_supported: The provision supports the requirement to report immediately and no later than two days for a widespread infringement or a serious incident as defined in Article 3(49)(b). However, the specific definition of that incident involving critical infrastructure disruption is not contained in the provided text. |
| a25 | dev | answer | answered | ai_act:art72 | ai_act:art72 | judge: partially_supported: The claim accurately reflects the core obligation to collect and analyse performance data, but it omits the exception that this obligation does not cover sensitive operational data from law-enforcement deployers. |
| a28 | test | answer | answered | ai_act:art50 | ai_act:art50 | judge: partially_supported: The claim accurately reflects the general obligation to disclose deep fakes, but it omits the exception for law enforcement and the specific limitations for artistic, creative, and satirical works. |
| a33 | dev | answer | answered | ai_act:anxIII | ai_act:anxIII | judge: partially_supported: The claim accurately states that remote biometric identification systems are included under Annex III, but it omits the exception for AI systems used solely for biometric verification to confirm a person's claimed identity.; judge: partially_supported: While the provision does list AI systems intended for emotion recognition, the claim omits the important condition that this applies only "in so far as their use is permitted under relevant Union or national law". |
