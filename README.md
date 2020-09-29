# Awesome-Papers
:bulb: To be AI Researcher
- 내가 관심있는 연구분야의 SOTA 및 최신 연구 동향을 계속 조사하다 보면
- 나만의 착안점, 개선할 점, contribution할 부분이 보일테니
- 그 때 따라갈게

## :question: Objective of `jinmang2/Awesome-Papers` Repo.
```
To be AI Researcher, Artist and Good Person...!!
```

## :office: NLP

### Tokenization
- [x] **BPE(Byte-Pair-Encoding)**; *A New Algorithm for Data Compression* (C-user journal 1994) [paper](https://www.derczynski.com/papers/archive/BPE_Gage.pdf)
  - [In Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding#cite_note-4)
- [x] **Adjust BPE on NMT**; *Neural Machine Translation of Rare Words with Subword Units* (ACL 2016) [paper](https://www.aclweb.org/anthology/P16-1162.pdf)
  - Compare between `n-gram` and `byte-pair-encoding`
  
Wordpiece

SentencePiece

Morphological

### Word Vector Representation
- [x] **NPLM**; *A Neural Probabilistic Language Model* (jmlr 2003) [paper](https://jmlr.org/papers/volume3/tmp/bengio03a.pdf)
  - NPLM's Reference -> 문장에서 단어의 역할을 학습
    - [ ] *Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks* (NIPS 2000) [paper](https://papers.nips.cc/paper/1679-modeling-high-dimensional-discrete-data-with-multi-layer-neural-networks.pdf)
      - NN으로 고차원 이진 분산 표현을 실시하는 아이디어 제시
    - [ ] *Extracting distributed representations of concepts and relations from positive and negative propositions* (IEEE 2000) [link](https://ieeexplore.ieee.org/document/857906)
      - Hinton 교수의 연구가 성공적으로 적용된 사례
    - [ ] *Natural Language Processing With Modular Pdp Networks and Distributed Lexicon* (Cognitive Science 1991 July) [link](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1503_2)
      - Neural network를 LM에 적용시키려 한 사례
  - NPLM's Reference -> word sequence distribution의 statistical model을 학습
    - [ ] *Sequential neural text compression* (IEEE 1996) [link](https://pubmed.ncbi.nlm.nih.gov/18255564/)
      - I Love Schmidhuber a lot :)
- [x] **Word2Vec 2013a**; *Efficient Estimation of Word Representations in Vector Space* (ICLR 2013) [paper](https://arxiv.org/pdf/1301.3781.pdf)
  - Introduce `Skip-Gram` & `CBOW`
  - Google Team
- [x] **Word2Vec 2013b**; *Distributed Representations of Words and Phrases and their Compositionality* (NIPS 2013) [paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Propose train optimization method such as `negative sampling`
- [ ] **GloVe(Global Word Vectors)**; *GloVe: Global Vectors for Word Representation* (ACL 2014) [paper](https://nlp.stanford.edu/pubs/glove.pdf)
  - Stanford Univ.
  - Overcome `Word2Vec` and `LSA`
- [ ] **Swivel(Submatrix-Wise Vector Embedding Learner)**; *Swivel: Improving Embeddings by Noticing What’s Missing* () [paper](https://arxiv.org/pdf/1602.02215.pdf)
  - Google, [source code](https://github.com/src-d/tensorflow-swivel/blob/master/swivel.py)
- [x] **FastText**; *Enriching Word Vectors with Subword Information* (17.06.16, arxiv) [paper](https://arxiv.org/pdf/1607.04606.pdf)

### NLP Tasks

A large annotated corpus for learning natural language inference, Bowman et al., 2015 (EMNLP)

A board-coverage challenge corpus for sentence understanding through inference, Williams et al., 2018

SQuad: 100,000+ questions for machine comprehension of text, Rajpurkar et al., 2016

introduction to th conll-2003 shared task: language-independent named entity recognition, Tjong Kim Sang and De Meulder, 2003

#### Neural Machine Translation
- [ ] **MRT(Minimum Risk Training)**; *Minimum Risk Training for Neural Machine Translation* (ACL 2016) [paper](https://www.aclweb.org/anthology/P16-1159.pdf)

#### Text Classification
- [x] **FastText for classification**; *Bag of Tricks for Efficient Text Classification* (ACL 2017) [link](https://www.aclweb.org/anthology/E17-2068/)
- [ ] **UNMFit**; *Universal Language Model Fine-tuning for Text Classification* (18.05.23, arxiv) [paper](https://arxiv.org/pdf/1801.06146.pdf)

#### Question Answering
Stochastic Answer Networks for Machine Reading Comprehension https://arxiv.org/abs/1712.03556

#### Textual Entailment
Enhanced LSTM for Natural Language Inference https://arxiv.org/abs/1609.06038

#### Semantic Role Labeling
Deep Semantic Role Labeling: What Works and What’s Next https://www.aclweb.org/anthology/P17-1044/

#### Summarization
>**Extractive**
- [ ] **BertSum**; *Fine-tune BERT for Extractive Summarization* (19.03.25, arxiv) [paper](https://arxiv.org/pdf/1903.10318.pdf)
- [ ] **BertSum-Full Paper**; *Text Summarization with Pretrained Encoders* (19.08.22, arxiv) [paper](https://arxiv.org/pdf/1908.08345.pdf)

### Pre-trained NLP Architecture


- [ ] *Semi-supervised sequence learning* (NIPS 2015) [paper](https://papers.nips.cc/paper/5949-semi-supervised-sequence-learning.pdf)

Word Representations: A Simple and General Method for Semi-Supervised Learning

| institute                   | subtitle                 | title                                                                                                               | journal | published | etc                                                                                                                              |
|-----------------------------|--------------------------|---------------------------------------------------------------------------------------------------------------------|---------|-----------|----------------------------------------------------------------------------------------------------------------------------------|
| AllenAI                     | ELMo                     | *Deep contextualized word representations*                                                                          | ACL     | 2018      | [paper](https://arxiv.org/pdf/1802.05365.pdf)                                                                                    |
| AllenAI                     | LongFormer               | *Longformer: The Long-Document Transformer*                                                                         | arxiv   | 20.04.10  | [paper](https://arxiv.org/pdf/2004.05150.pdf)                                                                                    |
| GoogleAI                    | BERT                     | *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*                                  | ACL     | 2018      | [paper](https://www.aclweb.org/anthology/N19-1423.pdf)                                                                           |
| GoogleAI                    | ALBERT                   | *ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS*                                      | ICLR    | 19.09.26  | [paper](https://arxiv.org/pdf/1909.11942.pdf)                                                                                    |
| GoogleAI                    | T5                       | *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*                                 | JMLR    | 19.10.23  | [paper](https://arxiv.org/pdf/1910.10683.pdf)                                                                                    |
| GoogleAI                    | PEGASUS                  | *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization*                                  | ICML    | 2020      | [paper](https://arxiv.org/pdf/1912.08777.pdf)                                                                                    |
| GoogleAI                    | ELECTRA                  | *ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS*                                      | ICLR    | 2020      | [paper](https://openreview.net/pdf?id=r1xMH1BtvB)                                                                                |
| DeepMind                    | Compressive Transformers | *COMPRESSIVE TRANSFORMERS FOR LONG-RANGE SEQUENCE MODELLING*                                                        | arxiv   | 19.11.13  | [paper](https://arxiv.org/pdf/1911.05507.pdf)                                                                                    |
| UNC Chapel Hill             | LXMERT                   | *LXMERT: Learning Cross-Modality Encoder Representations from Transformers*                                         | arxiv   | 19.08.20  | [paper](https://arxiv.org/pdf/1908.07490.pdf)                                                                                    |
| OpenAI                      | GPT-1                    | *Improving language understanding with unsupervised learning*                                                       | OpenAI  | 2018      | [paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| OpenAI                      | GPT-2                    | *Language Models are Unsupervised Multitask Learners*                                                               | OpenAI  | 2019      | [paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)                   |
| OpenAI                      | GPT-3                    | *Language Models are Few-Shot Learners*                                                                             | OpenAI  | 2020      | [paper](https://arxiv.org/pdf/2005.14165.pdf)                                                                                    |
| FAIR                        | FastText                 | *Advances in Pre-Training Distributed Word Representations*                                                         | arxiv   | 17.12.26  | [paper](https://arxiv.org/pdf/1712.09405.pdf)                                                                                    |
| FAIR                        | XLM                      | *Cross-lingual Language Model Pretraining*                                                                          | arxiv   | 19.01.22  | [paper](https://arxiv.org/pdf/1901.07291.pdf)                                                                                    |
| FAIR                        | FSMT                     | *Facebook FAIR's WMT19 News Translation Task Submission*                                                            | arxiv   | 19.07.15  | [paper](https://arxiv.org/pdf/1907.06616.pdf)                                                                                    |
| FAIR                        | RoBERTa                  | *RoBERTa: A Robustly Optimized BERT Pretraining Approach*                                                           | arxiv   | 19.07.26  | [paper](https://arxiv.org/pdf/1907.11692.pdf)                                                                                    |
| FAIR                        | MMBT                     | *Supervised Multimodal Bitransformers for Classifying Images and Text*                                              | arxiv   | 19.09.06  | [paper](https://arxiv.org/pdf/1909.02950.pdf)                                                                                    |
| FAIR                        | BART                     | *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension* | arxiv   | 19.10.29  | [paper](https://arxiv.org/pdf/1910.13461.pdf)                                                                                    |
| FAIR                        | CamemBERT                | *CamemBERT: a Tasty French Language Model*                                                                          | arxiv   | 19.11.10  | [paper](https://arxiv.org/pdf/1911.03894.pdf)                                                                                    |
| FAIR                        | mBART                    | *Multilingual Denoising Pre-training for Neural Machine Translation*                                                | arxiv   | 20.01.22  | [paper](https://arxiv.org/pdf/2001.08210.pdf)                                                                                    |
| FAIR                        | RAG                      | *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*                                                  | arxiv   | 20.05.22  | [paper](https://arxiv.org/pdf/2005.11401.pdf)                                                                                    |
| Hugging Face                | DistilBERT               | *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*                                     | arxiv   | 19.10.02  | [paper](https://arxiv.org/pdf/1910.01108.pdf)                                                                                    |
| Microsoft                   | Marian                   | *Marian: Cost-effective High-Quality Neural Machine Translation in C++*                                             | ACL     | 2018      | [paper](https://www.aclweb.org/anthology/W18-2716.pdf)                                                                           |
| Microsoft                   | MT-DNN                   | *Multi-Task Deep Neural Networks for Natural Language Understanding*                                                | arxiv   | 19.05.30  | [paper](https://arxiv.org/pdf/1901.11504.pdf)                                                                                    |
| Microsoft                   | LayoutLM                 | *LayoutLM: Pre-training of Text and Layout for Document Image Understanding*                                        | arxiv   | 19.12.31  | [paper](https://arxiv.org/pdf/1912.13318.pdf)                                                                                    |
| NVIDIA                      | MegatronLM               | *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*                             | arxiv   | 19.09.17  | [paper](https://arxiv.org/pdf/1909.08053.pdf)                                                                                    |
| Univ. of Washington         | Grover-Mega              | *Defending Against Neural Fake News*                                                                                | arxiv   | 19.10.29  | [paper](https://arxiv.org/pdf/1905.12616.pdf)                                                                                    |
| Carnegie Mellon GoogleBrain | Transformer-XL           | *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*                                           | arxiv   | 19.06.02  | [paper](https://arxiv.org/pdf/1901.02860.pdf)                                                                                    |
| Carnegie Mellon GoogleBrain | XLNet                    | *XLNet: Generalized Autoregressive Pretraining for Language Understanding*                                          | arxiv   | 19.06.19  | [paper](https://arxiv.org/pdf/1906.08237.pdf)                                                                                    |
| Carnegie Mellon GoogleBrain | Funnel                   | *Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing*                         | arxiv   | 20.06.05  | [paper](https://arxiv.org/pdf/2006.03236.pdf)                                                                                    |
| Salesforce                  | CTRL                     | *CTRL: A CONDITIONAL TRANSFORMER LANGUAGE MODEL FOR CONTROLLABLE GENERATION*                                        | arxiv   | 19.09.11  | [paper](https://arxiv.org/pdf/1909.05858.pdf)                                                                                    |
| Anonymous authors           | MobileBERT               | *MobileBERT: Task-Agnostic Compression of BERT by Progressive Knowledge Transfer*                                   | ICLR    | 2020      | [paper](https://openreview.net/pdf?id=SJxjVaNKwB)                                                                                |

## :sparkles: Attention Mechanism
- [x] **Bahdanau Attention**; *NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE* (ICLR 2015) [paper](https://arxiv.org/pdf/1409.0473.pdf)

- [x] **Multi-Head Attention**; *Attention Is All You Needs* (NIPS 2017) [paper](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

- [ ] **Google Research-Synthesizer**; *SYNTHESIZER: Rethinking Self-Attention in Transformer Models* (20.05.02, arxiv) [paper](https://arxiv.org/pdf/2005.00743.pdf)

## :massage: Conversational AI

### Memory-Based Research
- `Sumit Chopra`, `Jason Weston`님 연구 추적
- [x] *Memory Networks* (14.10.15, arxiv; ICLR 2015) [paper](https://arxiv.org/pdf/1410.3916.pdf)
- [x] *End-To-End Memory Networks* (NIPS 2015) [paper](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
- [ ] *Learning Through Dialogue Interactions By Asking Questions* (16.12.15, ICLR 2017) [paper](https://arxiv.org/pdf/1612.04936.pdf)

### Open-Domain
- Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index, ACL
- Kelvin Guu의 REALM, ACL
- [ ] **DPR**; *Dense Passage Retrieval for Open-Domain Question Answering* (20.04.10) [paper](https://arxiv.org/pdf/2004.04906.pdf)
  - [Huffon님 소개자료](https://www.facebook.com/111809756917564/posts/276190540479484/) 

## :art: Generative Model

### GAN
- [ ] **Original GAN**; *Generative Adversarial Net* (NIPS 2014) [paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

## :monkey_face: Meta Learning

- [ ] **MAML**; *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks* (ICML 2017) [paper](https://arxiv.org/pdf/1703.03400.pdf)

### Curiosity Algorithms
- https://ai.googleblog.com/2018/10/curiosity-and-procrastination-in.html
- [ ] Meta-leraning curiosity algorithms
- [ ] Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML
- [ ] Novelty search (Lehman & Stanley, 2008)
- [ ] Buffers and Nearest Neighbors (Fu et al., 2017)
- [ ] Generating goals (Srivastava et al., 2013; Kulkarni et al., 2016)
- [ ] Learning progress (Oudeyer et al., 2007; Schmidhuber, 2008)
- [ ] Generating diverse skills (Eysenbach et al., 2018)
- [ ] Stochastic neural networks (Florensa et al., 2017; Fortunato et al., 2017)
- [ ] Count-based exploration (Tang et al., 2017)
- [ ] Object-based curiosity measures (Forestier & Oudeyer, 2016)
- [ ] Bonus-based (Taiga et al., 2019)

### Road to General Intelligence
- AutoML Style Approach
  - Neural Architecture Search (NAS)
  - Hyperparameter optimization for deep networks
  - Auto-sklearn, Learning loss funtions to replace cross-entropy for training a fixed architecture on MNIST and CIFAR
- Meta-learning with genetic programming, evolutionary computing
- Programming Automation
  - Searching over mathematical operations within neural networks
  - Neural networks that learn programs
- Modular Meta-Learning / Hierarchical Meta-Learning, Reinforcement Learning
- Inspired from Cognitive/Brain Science (Attention, Curiosity, Common Sense, etc)
- Agent57 (DeepMind)


## :brain: Reinforcement Learning

### Theoretical
- [x] **Policy Gradient Theorem** *Policy Gradient Methods for Reinforcement Learning with Function Approximation* (NIPS 2000) [paper](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

### RL.start() 오늘의 논문 series
- [ ] *ACCELERATED METHODS FOR DEEP REINFORCEMENT LEARNING* () [paper](https://arxiv.org/pdf/1803.02811.pdf)
- [ ] *Implementation Matters In Deep RL* () [paper](https://openreview.net/forum?id=r1etN1rtPB)
- [ ] *CURL: Contrastive Unsupervised Representations for Reinforcement Learning* () [paper](https://arxiv.org/pdf/2004.04136)
- [ ] *Dream to Control: Learning Behaviors by Latent Imagination* () [paper](https://arxiv.org/abs/1912.01603)

### :information_source: DeepMind
**```project: ShallowMinded```**

- [x] 20.09.15; Deep Reinforcement Learning (16.06.17) [post](https://deepmind.com/blog/article/deep-reinforcement-learning)
  - Deep Q-Network for atari
    - [ ] **DQN**; *Playing Atari with Deep Reinforcement Learning* (13.12.19, arxiv) [paper](https://arxiv.org/pdf/1312.5602.pdf)
    - [ ] **DQN**; *Human-level control through deep reinforcement learning* (15.02.26, Nature) [paper](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
    - Improved the DQN methods as follows;
      - **Learning Dynamics**;
        - [ ] *Deep Reinforcement Learning with Double Q-learning* (15.12.08, arxiv) [paper](https://arxiv.org/pdf/1509.06461.pdf)
        - [ ] *Increasing the Action Gap: New Operators for Reinforcement Learning* (15.12.05, arxiv) [paper](https://arxiv.org/pdf/1512.04860.pdf)
      - [ ] **Replayed Experiences**; *PRIORITIZED EXPERIENCE REPLAY* (16.02.25, arxiv) [paper](https://arxiv.org/pdf/1511.05952.pdf)
      - [ ] **Normalising**; *Dueling Network Architectures for Deep Reinforcement Learning* (16.04.05, arxiv) [paper](https://arxiv.org/pdf/1511.06581.pdf)
      - [ ] **Aggregating**; *Deep Exploration via Bootstrapped DQN* (16.07.04, arxiv) [paper](https://arxiv.org/pdf/1602.04621.pdf)
      - [ ] **Re-scaling**; *Learning values across many orders of magnitude* (16.08.16, arxiv) [paper](https://arxiv.org/pdf/1602.07714.pdf)
    - Training a single NN to learn about multiple atari games
      - [ ] **Single NN for value function**; *Universal Value Function Approximators* (ICML 2015) [paper](http://proceedings.mlr.press/v37/schaul15.pdf)
      - [ ] **about Multiple Atari games**; *POLICY DISTILLATION* (Under review ICLR 2016; 16.01.07 arxiv) [paper](https://arxiv.org/pdf/1511.06295.pdf)
    - Gorila, massively distributed deep RL system
      - [ ] *Massively Parallel Methods for Deep Reinforcement Learning* (ICML 2015) [paper](https://8109f4a4-a-62cb3a1a-s-sites.googlegroups.com/site/deeplearning2015/1.pdf?amp%3Battredirects=2&attachauth=ANoY7cqdiRmUC8LKJASTJoOMnTGNSRtIhG1e7n_wRxRTHkS8QZekJj07Erf1988oVwV60t1dQ3Z0arm9jMjeJwv4RekhKlHkhoFlM6hwgqZufEug7XRMDdp4Qa1F620-j3DU_HX5Z4QT5g5g6c-vlKfLfzhtJzjCrabcDcn-4P640DmBrekqxSGOCMtx9imgraW22ZAteUc_4gMhWAzXzaFaZmQPxuZOTQ%3D%3D&attredirects=1)
  - Asynchronous RL approach
    ```
    exploits the multithreading capabilities of standard CPUs
    ```
    - [ ] **A3C (Asynchronosu Advantage Actor-Critic)**; *Asynchronous Methods for Deep Reinforcement Learning* (16.06.16, arxiv) [paper](https://arxiv.org/pdf/1602.01783.pdf)
      - [ ] **Intrinsic Motivation**; *Unifying Count-Based Exploration and Intrinsic Motivation* (16.11.07, arxiv) [paper](https://arxiv.org/pdf/1606.01868.pdf)
      - [ ] **Temporally abstract planning**; *Strategic Attentive Writer for Learning Macro-Actions* (16.06.15, arxiv) [paper](https://arxiv.org/pdf/1606.04695.pdf)
      - [ ] **Alternative approach based on episodic memory**; *Model-Free Episodic Control* (16.06.14, arxiv) [paper](https://arxiv.org/pdf/1606.04460.pdf)
      
  - Developed deep RL for continuous control problems
    - [ ] **DPG**; *Deterministic Policy Gradient Algorithms* (ICML 2014) [paper](http://proceedings.mlr.press/v32/silver14.pdf)
      ```
      provides a continuous analogue to DQN
      exploiting the differentiability of the Q-network to solve wide variety
      ```
      - [ ] **wide**; *CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING* (ICLR 2016, 19.07.05 arxiv) [paper](https://arxiv.org/pdf/1509.02971.pdf)
      - [ ] **variety**; *Learning Continuous Control Policies by Stochastic Value Gradients* (15.10.30, arxiv) [paper](https://arxiv.org/pdf/1510.09142.pdf)
      
  - [ ] **Asynchronous RL**; *Asynchronous Methods for Deep Reinforcement Learning* (ICML 2016) [paper](https://arxiv.org/pdf/1602.01783.pdf)
  
  - [ ] **AlphaGo**; *Mastering the game of Go with deep neural networks and tree search* (Nature 16.01.27) [link](https://www.nature.com/articles/nature16961)
  
  - Game theoretic approaches to deep RL
    - [ ] *Fictitious Self-Play in Extensive-Form Games* (ICML, 2015) [paper](http://proceedings.mlr.press/v37/heinrich15.pdf)
    - [ ] *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games* (16.06.28, arxiv) [paper](https://arxiv.org/pdf/1603.01121.pdf)
    - [ ] **culminating in a super-human poker player**; *Smooth UCT Search in Computer Poker* (IJCAI, 2015) [paper](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/view/11230/10741)
    
- [x] 20.09.16;	Announcing DeepMind Health research partnership with Moorfields Eye Hospital (16.07.05) [post](https://deepmind.com/blog/announcements/announcing-deepmind-health-research-partnership-moorfields-eye-hospital)

- [x] 20.09.17;	DeepMind AI Reduces Google Data Centre Cooling Bill by 40% (16.07.20) [post](https://deepmind.com/blog/article/deepmind-ai-reduces-google-data-centre-cooling-bill-40)
  - Read these posting together :) [link1](https://blog.google/outreach-initiatives/environment/data-centers-get-fit-on-efficiency/) [link2](https://blog.google/topics/environment/powering-internet-renewable-energy/) [link3](https://blog.google/topics/infrastructure/better-data-centers-through-machine/)
    - The Energy Efficiency Potential of Cloud-Based Software: A U.S. Case Study [link](https://crd.lbl.gov/assets/pubs_presos/ACS/cloud_efficiency_study.pdf)
   - [PUE Measure](https://www.google.com/about/datacenters/efficiency/)
   - [Machine Learning Applications for Data Center Optimization](https://docs.google.com/a/google.com/viewer?url=www.google.com/about/datacenters/efficiency/internal/assets/machine-learning-applicationsfor-datacenter-optimization-finalv2.pdf)

- [x] 20.09.18;	Decoupled Neural Interfaces Using Synthetic Gradients (16.08.29) [post](https://deepmind.com/blog/article/decoupled-neural-networks-using-synthetic-gradients)
  - [ ] **Synthesizer Gradient**; *Decoupled Neural Interfaces using Synthetic Gradients* (17.07.03, arxiv) [paper](https://arxiv.org/pdf/1608.05343.pdf)
  - This framework can also be thought about from an error critic point of view [Werbos](http://www.werbos.com/HICChapter13.pdf) and is similar in flavour to using a critic in reinforcement learning [Baxter](https://www.cis.upenn.edu/~mkearns/finread/BaxterWeaverBartlett.pdf)

- [x] 20.09.19;	Applying machine learning to radiotherapy planning for head & neck cancer (16.08.30) [post](https://deepmind.com/blog/announcements/applying-machine-learning-radiotherapy-planning-head-neck-cancer)

- [x] 20.09.21;	WaveNet: A generative model for raw audio (16.09.08) [post](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
  - [ ] **WaveNet**; *WAVENET: A GENERATIVE MODEL FOR RAW AUDIO* (16.09.19, arxiv) [paper](https://arxiv.org/pdf/1609.03499.pdf)
  - [Google Voice Search](https://ai.googleblog.com/2015/09/google-voice-search-faster-and-more.html)
  - [Speech Synthesis](https://en.wikipedia.org/wiki/Speech_synthesis)
  -  [ ] **Concatenative TTS**; *Unit selection in a concatenative speech synthesis system using a large speech database* (IEEE 96.05.09) [link](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Es-YRKMAAAAJ&citation_for_view=Es-YRKMAAAAJ:u5HHmVD_uO8C)
  - [ ] **Parametric TTS**; *Statistical parametric speech synthesis* (Elsevier Science Publishers BV 09.11.01) [link](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=z3IRvDwAAAAJ&citation_for_view=z3IRvDwAAAAJ:d1gkVwhDpl0C)
  - [Vocoder](https://en.wikipedia.org/wiki/Vocoder)
  - [ ] **PixelRNN**; *Pixel Recurrent Neural Networks* (16.08.19, arxiv) [paper](https://arxiv.org/pdf/1601.06759.pdf)
  - [ ] **PixelCNN**; *Conditional Image Generation with PixelCNN Decoders* (16.06.18, arxiv) [paper](https://arxiv.org/pdf/1606.05328.pdf)
  - [ ] **Google's current best `parametric` TTS**; *Fast, Compact, and High Quality LSTM-RNN Based Statistical Parametric Speech Synthesizers for Mobile Devices* (ISCA 2016) [link](https://research.google/pubs/pub45379/)
  - [ ] **Google's current best `concatenative` TTS**; *Recent Advances in Google Real-time HMM-driven Unit Selection Synthesizer* (ISCA 2016) [link](https://research.google/pubs/pub45564/)
  - [Mean Opinion Score Measure](https://en.wikipedia.org/wiki/Mean_opinion_score)
- [x] 20.09.22;	Putting patients at the heart of DeepMind Health (16.09.21) [post](https://deepmind.com/blog/announcements/putting-patients-heart-deepmind-health)
- [x] 20.09.23;	Announcing the Partnership on AI to Benefit People & Society (16.09.28) [post](https://deepmind.com/blog/announcements/announcing-partnership-ai-benefit-people-society)
- [x] 20.09.24;	Differentiable neural computers (16.10.12) [post](https://deepmind.com/blog/article/differentiable-neural-computers)
  - [ ] **DNC; Dynamic Neural Computer**; *Hybrid computing using a neural network with dynamic external memory* (nature 16.10.12) [link](https://www.nature.com/articles/nature20101) [paper](https://www.gwern.net/docs/rl/2016-graves.pdf)
  - [ ] **DNC; opinion piece**; *Deep neural reasoning* (Nature 16.10.12) [link](https://www.nature.com/articles/nature19477)
  - [ ] **NTM; Neural Turing Machine**; *Neural Turing Machines* (14.12.10) [paper](https://arxiv.org/pdf/1410.5401.pdf)
- [x] 20.09.25;	DeepMind and Blizzard to release StarCraft II as an AI research environment (16.11.04) [post](https://deepmind.com/blog/announcements/deepmind-and-blizzard-release-starcraft-ii-ai-research-environment)
- [ ] 20.09.26;	Reinforcement learning with unsupervised auxiliary tasks (16.11.17) [post](https://deepmind.com/blog/article/reinforcement-learning-unsupervised-auxiliary-tasks)
  - [ ] **UNREAL Agent**; *REINFORCEMENT LEARNING WITH UNSUPERVISED AUXILIARY TASKS* (16.11.16, arxiv) [paper](https://arxiv.org/pdf/1611.05397.pdf)
- [ ] 20.09.28; Working with the NHS to build lifesaving technology (16.11.22) [post](https://deepmind.com/blog/announcements/working-nhs-build-lifesaving-technology)
- [ ] 20.09.29; DeepMind Papers @ NIPS (Part 1) (16.12.02) [post](https://deepmind.com/blog/article/deepmind-papers-nips-part-1)
  - [ ] *Interaction Networks for Learning about Objects, Relations and Physics* (2016.12.01, arxiv; NIPS 2016) [paper](https://arxiv.org/pdf/1612.00222.pdf)
  - For applications of interaction networks to scene understanding and imagination-based decision-making,
    - [ ] *Discovering objects and their relations from entangled scene representations* (ICLR 2017) [paper](https://openreview.net/forum?id=rkrjrvmKl)
    - [ ] *Metacontrol for Adaptive Imagination-Based Optimization* (ICLR 2017) [paper](https://openreview.net/forum?id=Bk8BvDqex)
  - [ ] *Strategic Attentive Writer for Learning Macro-Actions* (; NIPS 2016)
  
  


## :chart_with_upwards_trend: Financial Mathematics & Engineer

## :cat2: Theoretical Deep Learning

- [x] *Neural Network Ensembles, Cross Validation, and Active Learning* (NIPS 1995) [paper](https://papers.nips.cc/paper/1001-neural-network-ensembles-cross-validation-and-active-learning.pdf)

Batch Normalization

Lipschitz gradient

Global Batch Normalization

Input Covariate Shift

Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

How Does Batch Normalization Help Optimization?

Layer Normalization https://arxiv.org/abs/1607.06450

LeCun Initialization [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

Xavier initialization [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

He Initialization [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf)

Nesterov Optimizer (Optimization류 논문들)

weight_standardization

## :heart_eyes: Schmidhuber
>[**Juergen Schmidhuber's Google Scholar**](https://scholar.google.co.kr/citations?user=gLnCTgIAAAAJ&hl=ko)
- [x] *Long short-term memory* (Neural Computation 1997) [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)
- [ ] *LSTM: A Search Space Odyssey* (IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS 2017) [paper](https://arxiv.org/pdf/1503.04069.pdf?fbclid=IwAR377Jhphz_xGSSThcqGUlAx8OJc_gU6Zwq8dABHOdS4WNOPRXA5LcHOjUg)
- [x] *Highway Networks* (15.05.03, arxiv) [paper](https://arxiv.org/pdf/1505.00387.pdf)
  - Full Paper: *Training Very Deep Networks* [link](https://arxiv.org/abs/1507.06228)
- [x] *Recurrent Highway Networks* (ICML 2017) [paper](http://proceedings.mlr.press/v70/zilly17a/zilly17a.pdf)
- [ ] *Gradient flow in recurrent nets: the difficulty of learning long-term dependencies* (IEEE 2001) [paper](https://ml.jku.at/publications/older/ch7.pdf) [paper](https://mediatum.ub.tum.de/doc/1290195/file.pdf)
- [ ] *Bidirectional LSTM networks for improved phoneme classification and recognition* (International Conference on Artificial Neural Networks 05.09.11) 
- [ ] *Sequential neural text compression* (IEEE 1996) [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.106.3159&rep=rep1&type=pdf)
- [ ] *Neural expectation maximazation* (NIPS 2017) [paper](https://papers.nips.cc/paper/7246-neural-expectation-maximization.pdf)
- [ ] *Accelerated Neural Evolution through Cooperatively Coevolved Synapses* (JMLR 2008) [paper](https://www.jmlr.org/papers/volume9/gomez08a/gomez08a.pdf)
- [ ] *World Models* (18.05.09, arxiv) [paper](https://arxiv.org/pdf/1803.10122.pdf)

## ETC

LSTM-SAE Unsupervised Pre-training of a Deep LSTM-based Stacked Autoencoder for Multivariate Time Series Forecasting Problems

C3D Learning Spatiotemporal Features with 3D Convolutional Networks

n-gram 관련 논문
- Estimation of Probabilities from Sparse Data for the
Language Model Component of a Speech Recognizer
- Interpolated estimation of Markov source parameters from sparse data

Pointing the Unknown Words (몬트리홀 대학)

Seq2Seq Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

Real-World Anomaly Detection in Surveillance Videos

self-attention on classification - A Structured Self-Attentive Sentence Embedding

