# Eval report

Generated 2026-11-08T19:34:22Z

Config: model=gemini:gemini-3.6-flash:thinking-low, embedder=hash-768, store=memory, scorer=judge:gemini:gemini-3.1-pro-preview, corpus_version=cbd4f9deaa0bc9cd

| scope | items | faithfulness | citation_precision | citation_hit_rate | refusal_accuracy | unsafe_answer_rate | false_refusal_rate | retrieval_recall |
|---|---|---|---|---|---|---|---|---|
| all | 87 | 0.872 | 0.936 | 0.984 | 0.954 | 0.000 | 0.060 | 0.970 |
| dev | 44 | 0.824 | 0.957 | 1.000 | 0.932 | 0.000 | 0.088 | 0.941 |
| test | 43 | 0.926 | 0.917 | 0.969 | 0.977 | 0.000 | 0.030 | 1.000 |
| ai_act | 35 | 0.867 | 0.886 | 0.971 | 1.000 | 0.000 | 0.000 | 1.000 |
| gdpr | 32 | 0.878 | 1.000 | 1.000 | 0.875 | 0.000 | 0.125 | 0.938 |
| out_of_scope | 20 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Gate

- FAIL: all.faithfulness = 0.8721 is below the floor 0.9
- FAIL: dev.faithfulness = 0.8242 is below the floor 0.9
- note: no baseline file: regression check skipped

## Items that were not fully right

| id | split | expected | got | cited | gold | issue |
|---|---|---|---|---|---|---|
| g03 | dev | answer | refused (model_declined) |  | gdpr:art6 | wrong refusal decision |
| g05 | dev | answer | answered | gdpr:art35 | gdpr:art35 | judge: partially_supported: The claim accurately captures the exception, but drops the specific conditions that the processing must be pursuant to Article 6(1)(c) or (e) and that the Union or Member State law must regulate the specific processing operation or set of operations in question. |
| g07 | dev | answer | refused (model_declined) |  | gdpr:art83 | wrong refusal decision |
| g09 | dev | answer | answered | gdpr:art13 | gdpr:art13 | judge: partially_supported: The provision requires the controller to provide the purposes and legal basis for processing, but the claim omits the condition that this specific article applies when data are collected from the data subject, as well as the exception if the data subject already has the information.; judge: partially_supported: The claim accurately reflects the requirement to inform the data subject about legitimate interests pursued, but omits the condition in Article 13(1) that this applies when personal data are collected from the data subject.; judge: partially_supported: The claim accurately states the requirement to provide information about recipients, but omits the condition in Article 13(1) that this applies specifically when personal data are collected from the data subject. |
| g13 | dev | answer | answered | gdpr:art5 | gdpr:art5 | judge: partially_supported: The claim accurately states the general rule for storage limitation but omits the provision's exception that allows personal data to be stored for longer periods for archiving in the public interest, scientific or historical research, or statistical purposes. |
| g15 | dev | answer | answered | gdpr:art30 | gdpr:art30 | judge: partially_supported: The claim accurately states the general obligation for controllers to maintain a record of processing activities, but it omits the exception for enterprises with fewer than 250 employees set out in paragraph 5. |
| g18 | test | answer | answered | gdpr:art34 | gdpr:art34 | judge: partially_supported: The claim correctly states the general obligation to communicate a high-risk data breach as set out in paragraph 1, but it omits the important exceptions in paragraph 3 under which this communication is not required. |
| g19 | dev | answer | answered | gdpr:art8 | gdpr:art8 | judge: partially_supported: The claim omits the crucial condition that this applies specifically when processing is based on consent ('Where point (a) of Article 6(1) applies').; judge: partially_supported: The claim states a general rule for processing based on consent for a child under 16, but the provision specifically limits this rule to the offer of information society services directly to a child. |
| g25 | dev | answer | refused (model_declined) |  | gdpr:art3 | wrong refusal decision |
| g28 | test | answer | refused (unverified) |  | gdpr:art39 | wrong refusal decision |
| a02 | test | answer | answered | ai_act:art50 | ai_act:art50 | judge: partially_supported: The claim omits important exceptions to the transparency obligation, specifically that the obligation does not apply if it is already obvious to a well-informed natural person or if the AI system is authorised by law for certain criminal justice purposes. |
| a04 | test | answer | answered | ai_act:art6 | ai_act:art6, ai_act:anxIII | judge: partially_supported: The claim correctly states the general rule from paragraph 2 that AI systems in Annex III are considered high-risk, but it omits the important exception in paragraph 3, which states they are not high-risk if they do not pose a significant risk of harm. |
| a05 | dev | answer | answered | ai_act:art26, ai_act:art27 | ai_act:art26 | imprecise citation |
| a06 | test | answer | answered | ai_act:art27 | ai_act:art27 | judge: partially_supported: The claim correctly identifies that public bodies and private entities providing public services must perform a fundamental rights impact assessment, but it drops the fact that this duty also belongs to deployers of high-risk AI systems referred to in points 5 (b) and (c) of Annex III. |
| a08 | test | answer | answered | ai_act:art9 | ai_act:art9 | judge: partially_supported: The claim omits that the provision specifically applies to providers of "high-risk AI systems", stating as general what the provision limits to certain cases. |
| a12 | test | answer | answered | ai_act:art15, ai_act:art13, ai_act:art42 | ai_act:art15 | imprecise citation |
| a13 | dev | answer | answered | ai_act:art53, ai_act:art54, ai_act:art55 | ai_act:art53, ai_act:art55 | imprecise citation |
| a14 | test | answer | answered | ai_act:art51, ai_act:art52 | ai_act:art51, ai_act:anxXIII | imprecise citation |
| a15 | dev | answer | answered | ai_act:art99, ai_act:art100 | ai_act:art99 | imprecise citation |
| a17 | dev | answer | answered | ai_act:art3 | ai_act:art3 | judge: partially_supported: The claim correctly restates most of the definition of an AI system, but it omits the final condition that the generated outputs must be able to influence physical or virtual environments. |
| a18 | test | answer | answered | ai_act:art3 | ai_act:art57, ai_act:art58 | imprecise citation |
| a25 | dev | answer | answered | ai_act:art72 | ai_act:art72 | judge: partially_supported: The claim accurately captures the main obligations of the post-market monitoring system to collect, document, and analyze data for compliance. However, it omits the exception that this obligation does not cover the sensitive operational data of deployers that are law-enforcement authorities. |
| a28 | test | answer | answered | ai_act:art50 | ai_act:art50 | judge: partially_supported: The claim accurately reflects the general obligation for deployers of AI systems generating deep fakes to disclose the artificial nature of the content. However, it omits important exceptions and limitations provided in Article 50(4), such as when the use is authorised by law for criminal investigations or when the content is part of an artistic or creative work. |
| a31 | dev | answer | answered | ai_act:art5 | ai_act:art5 | judge: partially_supported: The claim correctly states the general prohibition on AI systems inferring emotions in the workplace, but it omits the important exception for systems intended for medical or safety reasons. |
| a33 | dev | answer | answered | ai_act:anxIII | ai_act:anxIII | judge: partially_supported: The claim correctly states that remote biometric identification systems are included, but it omits the exception for AI systems used solely for biometric verification to confirm a person's claimed identity.; judge: partially_supported: The claim accurately quotes the type of AI systems included under the biometrics area, but omits the condition that these systems are only included "in so far as their use is permitted under relevant Union or national law". |
| a34 | test | answer | answered | ai_act:art13, ai_act:art15 | ai_act:art13 | imprecise citation |
| a35 | dev | answer | answered | ai_act:art2 | ai_act:art2 | judge: partially_supported: The provision explicitly states that the Regulation shall not affect Regulation (EU) 2016/679, but specifies the exceptions as Article 10(5) and Article 59. The claim generalizes these specific exceptions to "specific provisions on data governance", which loses important precision and may not accurately reflect the content of Article 59. |
