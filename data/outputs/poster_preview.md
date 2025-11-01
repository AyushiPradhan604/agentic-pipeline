# Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum Hadi Pouransari1,◦

Hadi Pouransari1,◦, Vaishaal Shankar2,†

## Abstract

- Large language models (LLMs) are commonly trained on datasets consisting of fixed-length token sequences.
- These datasets are created by randomly concatenating documents of various lengths and then chunking them into sequences of a predetermined target length (concat-and-chunk).
- In this study, we introduce dataset decomposition, a novel variable sequence length training technique, to tackle these challenges.
- We decompose a dataset into a union of buckets, each containing sequences of the same size extracted from a unique document.
- We train an 8k context-length 1B model at the same cost as a 2k context-length model trained with the baseline approach.
- Experiments on a webscale corpus demonstrate that our approach significantly enhances performance on standard language evaluations and long-context benchmarks, reaching target accuracy with up to 6× faster training compared to the baseline.


## Introduction

- Large language models (LLMs) are often pretrained autoregressively (i.e., predicting the next token given a context) on large text corpora sourced from the web.
- Each of these datasets comprises multiple documents, ranging from Wikipedia articles to books and code repositories.
- In this paper, we investigate the influence of document chunking, propose alternative strategies, and evaluate the proposed strategies with careful experiments.
- *Code to be available at https://github.com/apple/ml-dataset-decomposition.
- 38th Conference on Neural Information Processing Systems (NeurIPS 2024).
- [[PAGE 2]] 233 234 235 236 237 238 239 240 # of seen tokens 40 45 Regular eval (%) Baseline-8k Dataset Decomposition >4⨉ data efﬁciency +2.4 (a) Data Efficiency 0 1000 2000 3000 4000 5000 GPU-Hours 40 45 Regular eval (%) Baseline-8k Dataset Decomposition >6⨉ training speed-up (b) Computational Effic


## results?

- Answer: [Yes] Justification: We provide all implementation details in Appendix B.
- Guidelines: • The answer NA means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- 7.
- Experiment Statistical Significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
- We repeat all experiments at 17B total tokens scale twice (with different random seeds) and report mean and variance in Section 3.2.


## Method

- 2.1 Dataset decomposition Given a dataset D of tokenized documents {d1, d2, .
- , dn}, the goal of dataset decomposition (DD) is to reorganize D as a union of buckets, ∪iDi, such that: (1) each bucket Di consists of sequences of tokens with length li; (2) each sequence s ∈Di is a subsequence of one document d ∈D; and (3) each token in D appears in exactly one Di.
- Dataset decomposition as defined above is not unique.
- We propose a specific decomposition, with li = 2i, to optimally maintain the original document sequence length distribution while also enabling efficient batch pretraining, as explained in Section 2.2.
- We apply decomposition at the document level, which makes it very easy to integrate the method into any existing data preparation pipeline (a stage before model training) and is scalable to large datasets.
- For a tokenized document d ∈D with length l, where l = 2i1 + 2i2 + .
![](data/outputs\images\img_p6_1_99b67464e74e.png)
![](data/outputs\images\img_p6_2_16926274f5f3.png)
![](data/outputs\images\img_p6_3_11d270ad5a16.png)
![](data/outputs\images\img_p6_xref72_16926274f5f3.png)
![](data/outputs\images\img_p6_xref71_99b67464e74e.png)
![](data/outputs\images\img_p6_xref77_11d270ad5a16.png)
![](data/outputs\images\img_p17_1_2152ef1c0d0c.png)
![](data/outputs\images\img_p17_xref106_2152ef1c0d0c.png)
![](data/outputs\images\img_p20_1_beda83b61ced.png)
![](data/outputs\images\img_p20_xref124_beda83b61ced.png)

