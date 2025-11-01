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


## Method

- 2.1 Dataset decomposition Given a dataset D of tokenized documents {d1, d2, .
- , dn}, the goal of dataset decomposition (DD) is to reorganize D as a union of buckets, ∪iDi, such that: (1) each bucket Di consists of sequences of tokens with length li; (2) each sequence s ∈Di is a subsequence of one document d ∈D; and (3) each token in D appears in exactly one Di.
- Dataset decomposition as defined above is not unique.
- We propose a specific decomposition, with li = 2i, to optimally maintain the original document sequence length distribution while also enabling efficient batch pretraining, as explained in Section 2.2.
- We apply decomposition at the document level, which makes it very easy to integrate the method into any existing data preparation pipeline (a stage before model training) and is scalable to large datasets.
- For a tokenized document d ∈D with length l, where l = 2i1 + 2i2 + .


## PDF

- 3.8% 0.3% 0.6% Orig.
- Docs mean=598 Baseline-2k mean =463 Baseline-8k mean =558 Pack-8k mean =585 (a) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 Bucket number 2 4 6 8 10 12 14 16 Tokens % 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 (b) 25 26 27 28 29 210 211 212 213 Context length 10−5 10−4 10−3


## PDF

- Baseline-8k mean=930 Pack-8k mean=1064


## DD≥256

- mean=1344 (c) Figure 3: For the RefinedWeb dataset : (a) Distribution of chunk lengths using different dataset preparation methods.
- Peaks show the percentage of chunks for each method with the same length as the target sequence length.
- Color/pattern shows the ⌊log2 l⌋, where l is the length of the original document each token is extracted from.
- (c) Probability distribution of context length (number of tokens from the same document a token can attend to) observed during training for the concat-and-chunk baseline with target sequence length 8192 and DD with ≥256 mixture defined in Table 1.
- of RefinedWeb dataset tokens over different buckets, where D9 (corresponding to sequences with length 512) has the maximum tokens.
- Most tokens in a bucket Di are extracted from documents with length l such that 2i ≤l < 2i+1, and some tokens are rolled over from documents with length l ≥2i+1.
![](data/outputs\images\img_p6_1_99b67464e74e.png)
![](data/outputs\images\img_p6_2_16926274f5f3.png)
![](data/outputs\images\img_p6_3_11d270ad5a16.png)
![](data/outputs\images\img_p6_xref72_16926274f5f3.png)
![](data/outputs\images\img_p6_xref71_99b67464e74e.png)
![](data/outputs\images\img_p6_xref77_11d270ad5a16.png)


## MDQA

- 


## D6

- 


## D7

- 


## D8

- 


## D9

- 


## D10

- 


## D11

- 


## D12

- 


## D13

- 


## CSR

- 


## LU

- 


## RC

- 


## WK

- Avg.
- 10 20 30 Avg.
- Natural 3 6 10 17 21 17 13 9 482 1018 244 62.4 65.4 43.8 43.9 54.0 26.7 20.7 18.5 21.9 Equal 12 12 12 12 12 12 12 12 257 1020 244 61.9 64.3 43.1 43.5 53.3 25.1 21.4 17.4 21.3 1k-only 0 0 0 0 96 0 0 0 1024 512 234 60.8 66.4 43.2 44.7 54.0 0.2 0.1 0.2 0.2 ≤2k 16 16 16 16 16 16 0 0 195 336 231 62.8 63.
- Each row corresponds to a model trained on a specific mixture of dataset decomposition buckets.
- All models are OpenLM-1B, have seen a total of 96 × 230 tokens, use RoPE with a base frequency of 10k, and are trained with the same hyperparameters.
- reduce statistical error, we train each model twice from scratch with different random seeds and report the average metric for each benchmark (observing an average standard deviation of ∼0.3 for regular benchmarks and ∼1.6 for multi-document QA).


## MDQA

- 


## D8

- 


## D9

- 


## D10

- 


## D11

- 


## D12

- 


## D13

- 


## CSR

- 


## LU

- 


## RC

- 


## WK

- Avg.
- 10 20 30 Avg.
- Uniform 1 1 1 1 1 1 1 62.2 65.2 43.4 44.0 53.8 27.3 22.0 19.6 23.0 Grow-Linear 6 5 4 3 2 1 1 60.9 64.2 46.6 42.9 53.6 30.9 26.0 23.9 26.9 8 62.7 65.0 45.4 44.7 54.5 30.1 25.3 22.8 26.1 Grow-P2 32 16 8 4 2 1 1 60.9 64.3 46.5 44.1 54.0 29.6 25.0 23.1 25.9 8 62.8 65.2 45.3 44.2 54.4 32.3 26.9 24.6 28.0
- All models are OpenLM-1B and have seen a total of 96 × 230 tokens, with exactly 234 tokens from each Di for i = 8, .
- , 13.
- We use RoPE with a base frequency of 100k and the same default hyperparameters.


## Method

- Num Time ∆ Regular ∆


## MDQA

- ∆ GPUs (hours) Avg.
- 160M Baseline-8k 16 18.3 39.3 9.7


## DD

- 15.7 -14% 40.0 +0.7 11.4 +1.7 410M Baseline-8k 16 38.9 48.3 14.8


## DD

- 29.6 -24% 49.4 +1.1 18.8 +4.0 1B Baseline-8k 32 44.4 56.7 25.6


## DD

- 35.4 -20% 58.4 +1.7 25.6 Table 3: Comparing baseline training with DD on an alternative pretraining dataset and model sizes.


## Method

- fb Regular


## MDQA

- Avg.
- Baseline-8k 10k 51.3 19.0 100k 51.5 24.4


## DD≥256

- 10k 53.8 20.1 100k 53.8 24.9 Table 4: Effect of RoPE base frequency, fb, in pretraining.
- Model scaling We report results on OpenLM-1B, -3B, and -7B trained from scratch for a total of 237 tokens in Fig.
- 4b.
- We compare baseline training with a fixed target sequence length 8192 and VSL training with a DD≥256 mixture and the "Grow-Linear" curriculum with 8 cycles.
- Training with DD results in significant accuracy gains and reductions in training wall-clock time at different scales.
- Alternative dataset We demonstrate the efficacy of our proposed method on another large-scale dataset, DataComp-LM .


## Method

- 


## CSR

- 


## LU

- 


## RC

- 


## WK

- Avg.


## MDQA

- 


## TOEFL

- QuALITY Avg.
- 10 20 30 Baseline-8k 60.6 62.5 41.5 41.3 51.5 29.0 23.8 20.5 26.2 32.0 27.5 304 $ Baseline-8k+DM 60.2 64.1 42.8 41.8 52.4 24.4 20.0 16.0 29.2 32.0 27.1 304 $ Pack-8k+DM 60.3 64.0 44.6 41.8 52.7 25.6 19.8 16.9 29.2 33.1 27.7 304 $$


## ICLM

- 60.6 62.1 44.7 40.0 51.7 26.7 20.0 22.0 28.7 34.6 28.7 304 $$$ DD (ours) 62.8 65.2 45.3 44.2 54.4 32.3 26.9 24.6 30.7 34.2 30.9 244 $ Table 5: Comparison with baseline and state-of-the-art methods.
- All models are trained with the same hyperparameters, RoPE with fb = 100k, and for 103B tokens.
- DD uses the "Grow-P2" curriculum with 8 cycles.
- Dataset preparation cost is symbolic to compare methods and does not reflect the wall-clock time.
- 4 Related works
- DM denotes training with document masking.


## Method

- Doc Masking Average Context Docs in a Sequence Curr.


## MDQA30 (%)

- Baseline ✓ 930 Mult-random ✗ 16.0 Pack8k+DM ✓ 1064 Mult-packing ✗ 16.9 DD-Uniform ✗ 1344 Single ✗ 19.6 Baseline ✗ 4096 Mult-random ✗ 20.5


## ICLM

- ✗ 4096 Mult-semantic ✗ 22.0 DD-Grow-P2


## N/A

- 1344 Single ✓ 24.6 Table 6: Summary of long-context performance for different methods from Table 2 and Table 5.
- For example, Llama3 , ICLM , and , which we discussed in Section 3.6.
- Related to our study on sequence length bias, shows the importance of train-vs-test time distribution shift from a sequence length perspective on a string editing task.
- GrowLength proposes accelerating LLM pretraining by progressively growing context length using the baseline sequence formation method, but does not show results on LLMs.
- Similarly, increasing sequence length has been shown in BERT model training to improve compute efficiency.
- Different from these works, in dataset decomposition, we do not simply put documents with similar lengths into the same bucket.


## URL

- https://github.com/mlfoundations/open_lm/.
- arXiv preprint arXiv:2308.16137, 2023.
- arXiv preprint arXiv:2310.00576, 2023.
- Advances in Neural Information Processing Systems, 36, 2024.
- arXiv preprint arXiv:2107.02027, 2021.
- Transactions of the Association for Computational Linguistics, 7:453-466, 2019.


## A

- Broader impacts This work enables faster training of LLMs, which are among the most compute-intensive applications in the field.
- While we did not directly measure this potential benefit, a concurrent work shows such a benefit when cross-document attention is not allowed during LLM pretraining.
- A positive societal/environmental impact of this work is training LLMs with a smaller carbon footprint.
- Another potential societal advantage of this work is training LLMs with fewer hallucinations.


## B

- Implementation details


## B.1

- Training details Software and hardware details All experiments in this paper are conducted using the OpenLM|| repository, which is based on PyTorch.
- We use Fully Sharded Data Parallelism (FSDP) with Bfloat16 mixed precision for all experiments.
- For hardware, we use one or more nodes of 8× NVIDIA H100 GPUs (Hopper architecture), each with 80GB memory, and 192 CPU cores with 2000GB of RAM.
- Model architecture details We provide details of all architectures used in the paper in Table 7 to Table 11.
- Model Name OpenLM-160M Hidden dimension 768 Number of Layers 12 Number of Heads 12 Number of Parameters 162,435,840 Table 7: OpenLM-160M.
- Model Name OpenLM-410M Hidden dimension 1024 Number of Layers 24 Number of Heads 16 Number of Parameters 411,665,408 Table 8: OpenLM-410M.
![](data/outputs\images\img_p17_1_2152ef1c0d0c.png)
![](data/outputs\images\img_p17_xref106_2152ef1c0d0c.png)


## B.2

- Length based sampling and curriculum algorithm We present the details of our length-based sampling and curriculum in Algorithm 1.
- Algorithm 1 Length based sampling and curriculum Require: • Di: list of buckets such that Di includes sequences with length 2i • ni: total number of tokens to be picked from each bucket (see Table 1) • oi: sampling odd for each bucket (see Table 2) • c: number of cycles • b: number of tokens per opt
- , c] do ▷loop over cycles while at least one si,j is non-empty do odds ←[oi if si,j is not empty else 0 for i = 1, 2, 3, .
- .] probs ←odds/odds.sum() randomly sample index i with probability probs[i] sample b/2i sequences from si,j w/o replacement for training end while end for


## B.3

- Evaluation details Multi Document Question Answering (MDQA) We follow the open-book evaluation setup described in .
- The document containing the answer is part of the context.
- The evaluation script provided by the official repository processes the model’s response by using only the text before the first occurrence of a newline character as the answer.
- We noticed that sometimes the model responds with multiple newline characters before providing any valid text.
- In view of this behavior, we updated the evaluation script to look for the first non-empty text output from the model instead of the first string after newline character.
- Apart from this change in processing the model output, the rest of the evaluation follows the official implementation .


## TOEFL

- We follow the setup described in .
- As described in Section 3, the dataset contains multiple-choice QA pairs for the 15 longest lectures in .
- The choice corresponding to 17 [[PAGE 18]] 64 128 256 512 1024 2048 4096 8192 16384 sequence length 1 2 4 8 16 32 64 128 256 number of sequences per batch 100 99 100 100 100 107 153 304 719 99 100 100 101 106 143 265 566 100 102 100 105 138 243 487 100 100 106 137 232 445 102 105 135 227 426 105 135
- After we obtain the response, the computation of accuracy follows the official implementation .
- The dataset contains long documents with each document containing multiple-choice QA pairs.
- Sometimes the context for a QA pair can be longer than 8192 tokens.


## C

- Additional results


## C.1

- Additional results for training efficiency We enumerate model sizes (OpenLM-1B, OpenLM-3B, OpenLM-7B), the number of sequences in a batch (from 1 to 256), and sequence lengths (26 to 214) and measure the time to train 100 batches.
- We repeat this 5 times and report the average and standard deviation time per batch in Fig.
- 7.
- Notice that in the figure, each diagonal corresponds to a fixed b (number of tokens seen per optimization step).


## C.2

- Additional results for sequence length bias experiments In this section, we show that changing hyperparameters does not alter our conclusions in Section 3.2.
- We observed that pretraining on a sequence length of 1024 results in optimal performance with respect to regular metrics, compared to both longer and shorter lengths.
- For example, the regular average metric is 48.0 when pretraining with a 1024 sequence length, but it is 47.0 when pretraining with a 2048 sequence length.
- We explore whether this gap can be filled by using potentially better hyperparameters when training with a 2048 sequence length.
- Results are shown in Table 13, 18 [[PAGE 19]] demonstrating that the gap cannot be simply filled by choosing a different hyperparameter and is fundamental to the choice of pretraining sequence length.
- Maximum Learning Rate RoPE fb Regular Average 3 × 10−3 10,000 47.0 3 × 10−3 100,000 47.1 10−3 10,000 45.9 10−2 10,000 46.5 Table 13: Sensitivity to hyperparameters for Section 3.2 experiments.


## C.3

- Additional results for scaling experiments In this section, we show additional results for the experiments presented in Section 3.5.
- Table 14 shows results for dataset scaling, Table 15 for model scaling, and Table 16 for experiments on an alternative dataset.


## Method

- Regular average MDQA average 234 Baseline-8k 45.2 7.8


## DD

- 47.0 16.0 235 Baseline-8k 47.6 15.4


## DD

- 50.6 23.3 236 Baseline-8k 50.2 19.9


## DD

- 52.1 22.3 237 Baseline-8k 51.9 23.2


## DD

- 54.9 25.9 238 Baseline-8k 53.6 25.8


## DD

- 56.0 29.4 Table 14: Dataset scaling for OpenLM-1B.


## Method

- Regular average MDQA average 1B Baseline-8k 51.9 23.1


## DD

- 54.9 24.2 3B Baseline-8k 57.5 17.8


## DD

- 59.0 31.1 7B Baseline-8k 59.8 31.7


## DD

- 62.5 34.7 Table 15: Model scaling for total of 137B tokens.


## Method

- 


## PIQA

- 


## COPA

- 


## OBQA

- LamOAI HelSwg WinG WinGE SQuaAD BoolQ CoQA Jeop ArcE ArcC WikiQA


## MDQA

- 0-shot 0-shot 10-shots 0-shot 0-shot 3-shots 5-shots 3-shots ,0-shot 0-shot 3-shots 3-shots 3-shots 3-shots 10 20 30 160M Baseline 66.5 61 29.2 40.5 37.2 63.4 51.9 12.9 55.6 18.2 2.3 49.5 25.9 36.2 12.8 9.3 7.1


## DD

- 66.4 66 30.2 43.6 37.7 66.3 52.2 14.3 50.7 19.1 4.3 51.5 24 34 16.2 9.9 8.2 410M Baseline 69.8 68 37.4 53.0 50.4 74.0 55.8 30.0 59.7 28.5 12.1 59 29.8 48.3 18.9 13.4 12


## DD

- 71.5 70 38 55.8 51.6 74.7 56.3 27 59.5 26.2 17.6 60.4 30.5 52.2 24.4 18.1 14 1B Baseline 74.9 74 43.4 63 62.7 80.2 63.4 41.8 64.1 35.3 29.7 65.7 38.4 56.7 31.3 24.8 20.8


## DD

- 76.7 75 42.6 64.7 64.7 82.8 65 41.8 66.4 38.3 32.6 68.4 39.8 58.7 31.6 24.7 20.4 Table 16: Small model performance trained on an improved refined-web pipeline applied to Common Crawl.
- All models are trained for a total of 237 tokens.


## D

- Comparison to best-fit sequence packing Some recent works have employed a bin packing-based strategy which aims to reduce document cross-attention by minimizing unnecessary document truncation.
- To achieve this, they implement a known approximation algorithm called best-fit decreasing, which packs document chunks into sequences as tightly as possible.
- To compare with our method, we created a new dataset based on our implementation of the best-fit decreasing algorithm and trained a new model using this dataset.
- We present our implementation of the best-fit decreasing algorithm, the dataset we created, and the model we trained for comparison.
- Given a dataset D, the input to the algorithm is a list of tokenized document chunks C = {c1, c2, .
- , cK} such that SK i=1 ci = D, where each chunk is at most context size n (e.g., 2048) in length.
![](data/outputs\images\img_p20_1_beda83b61ced.png)
![](data/outputs\images\img_p20_xref124_beda83b61ced.png)


## E

- Training stability with VSL and curriculum presents the stability-efficiency dilemma: efficient LLM pretraining with massive data parallelism results in a large batch size and requires a high learning rate.
- Here, we show that dataset decomposition alleviates this problem when used with a curriculum: starting training by sampling more from short sequence buckets.
- We empirically demonstrate this by training an OpenLM-1B model from scratch with a high learning rate (= 10−2) and no gradient clipping, once with baseline-8k and once with DD using the "Grow-P100" curriculum.
- Training loss is shown in Fig.
- 8, demonstrating the stability of training with DD in comparison to the baseline.
- Loss Baseline-8k DD-Grow-P100 Figure 8: We compare the training loss when training with Baseline-8k versus DD with the "GrowP100" curriculum.


## F

- Average sequence length vs average context length We compute the mean of length (Fig.
- 3a) and context (Fig.
- 3c) distributions as follows.
- Assume a list of sequences with lengths l1, l2, .
- , lN, which are, for example, the chunk lengths in the concat-and-chunk approach or the sequence lengths in different buckets of the dataset decomposition approach.
- We define the average sequence length as follows: Average sequence length = 1


## N

- 


## N

- 


## X

- i li (1) In auto-regressive training on a sequence with length l, we apply l losses for next-token prediction on each token in parallel.
- Hence, for a sequence with length l, we see contexts with lengths equal to 20 [[PAGE 21]] 0, 1, 2, .
- , l −1.
- We define the average context length, which is different from the average sequence length, as follows: Average context length =  


## N

- 


## X

- i=1 li−1


## X

- j=0 j  /


## N

- 


## X

- i=1 li !


## N

- 


## X

- i=1 li(li −1) !
- / 2


## N

- 


## X

- i=1 li !
- (2) In Fig.
- 3a, Fig.
- 3c, and Table 1, we report the average sequence length and average context length for original documents, concat-and-chunk, and dataset decomposition with different mixtures.
- 21 [[PAGE 22]] NeurIPS Paper Checklist 1.
- Answer: [Yes] Justification: Method proposed in Section 2.1 and experiments provided in Section 3 support all claims in the abstract.


## results?

- Answer: [Yes] Justification: We provide all implementation details in Appendix B.
- Guidelines: • The answer NA means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- 7.
- Experiment Statistical Significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
- We repeat all experiments at 17B total tokens scale twice (with different random seeds) and report mean and variance in Section 3.2.


## URL.

- • The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets.
- Their licensing guide can help determine the license of a dataset.
- • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- 13.
- • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates.

