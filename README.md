# Awesome-Papers
:bulb: To be AI Researcher

## :office: NLP

### Tokenization
- [x] **BPE(Byte-Pair-Encoding)**; *A New Algorithm for Data Compression* (C-user journal 1994) [paper](https://www.derczynski.com/papers/archive/BPE_Gage.pdf)
  - [In Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding#cite_note-4)
- [x] **Adjust BPE on NMT**; *Neural Machine Translation of Rare Words with Subword Units* (ACL 2016) [paper](https://www.aclweb.org/anthology/P16-1162.pdf)
  - Compare between `n-gram` and `byte-pair-encoding`

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

#### Text Classification
- [x] **FastText for classification**; *Bag of Tricks for Efficient Text Classification* (ACL 2017) [link](https://www.aclweb.org/anthology/E17-2068/)
- [ ] **UNMFit**; *Universal Language Model Fine-tuning for Text Classification* (18.05.23, arxiv) [paper](https://arxiv.org/pdf/1801.06146.pdf)

### Pre-trained NLP Architecture
- [ ] *Semi-supervised sequence learning* (NIPS 2015) [paper](https://papers.nips.cc/paper/5949-semi-supervised-sequence-learning.pdf)

#### AllenAI
- [x] **ELMo(Embeddings Language Model)**; *Deep contextualized word representations* (ACL 2018) [paper](https://arxiv.org/pdf/1802.05365.pdf)
  - [Introduction](https://allennlp.org/elmo)

#### GoogleAI
- [x] **BERT(Bidirectional Encoder Represenations from Transformer)**; (ACL 2018) [paper](https://www.aclweb.org/anthology/N19-1423.pdf)
- [ ] **ALBERT**; *ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS* (19.09.26, arxiv; ICLR 2020) [paper](https://arxiv.org/pdf/1909.11942.pdf)
- [ ] **ELECTRA**; *ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS* (ICLR 2020) [paper](https://openreview.net/pdf?id=r1xMH1BtvB)

#### OpenAI
- [ ] **GPT-1**; *Improving language understanding with unsupervised learning* (Technical report, OpenAI 2018) [paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [ ] **GPT-2**; *Language Models are Unsupervised Multitask Learners* (Technical report, OpenAI 2019) [paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ ] **GPT-3**; *Language Models are Few-Shot Learners* () [paper](https://arxiv.org/pdf/2005.14165.pdf)

#### Hugging Face
- [ ] **DistilBERT**; *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter* (19.10.02, arxiv) [paper](https://arxiv.org/pdf/1910.01108.pdf)

#### MicroSoft
- [ ] **MT-DNN**; *Multi-Task Deep Neural Networks for Natural Language Understanding* (19.05.30, arxiv) [paper](https://arxiv.org/pdf/1901.11504.pdf)

#### NVIDIA
- [ ] **MegatronLM**; *Megatron-LM: Training Multi-Billion Parameter Language Models Using
Model Parallelism* (19.09.17, arxiv) [paper](https://arxiv.org/pdf/1909.08053.pdf)

#### Univ. of Washington
- [ ] **Grover-Mega**; *Defending Against Neural Fake News* (19.10.29, arxiv) [paper](https://arxiv.org/pdf/1905.12616.pdf)

#### Carnegie Mellon University (with Google Brain)
- [ ] **Transformer-XL**; *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context* (19.06.02, arxiv) [paper](https://arxiv.org/pdf/1901.02860.pdf)
- [ ] **XLNet**; *XLNet: Generalized Autoregressive Pretraining for Language Understanding* (v1:19.06.19 , v2:20.01.02, arxiv) [paper](https://arxiv.org/pdf/1906.08237.pdf)
  - 19년도 6월에 발표한걸로 기억
  
#### Salesforce Research
- [ ] **CTRL**; *CTRL: A CONDITIONAL TRANSFORMER LANGUAGE MODEL FOR CONTROLLABLE GENERATION* (19.09.11, arxiv) [paper](https://arxiv.org/pdf/1909.05858.pdf)

#### FAIR
- [x] **FastText-completed**; *Advances in Pre-Training Distributed Word Representations* (17.12.26, arxiv) [paper](https://arxiv.org/pdf/1712.09405.pdf)
- [ ] **XLM**; *Cross-lingual Language Model Pretraining* (19.01.22, arxiv) [paper](https://arxiv.org/pdf/1901.07291.pdf)
- [ ] **RoBERTa**; *RoBERTa: A Robustly Optimized BERT Pretraining Approach* (19.07.26, arxiv) [paper](https://arxiv.org/pdf/1907.11692.pdf)
- [ ] **BART**; *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension* (19.10.29, arxiv) [paper](https://arxiv.org/pdf/1910.13461.pdf)
- [ ] **CamemBERT**; *CamemBERT: a Tasty French Language Model* (19.11.10, arxiv) [paper](https://arxiv.org/pdf/1911.03894.pdf)

## :sparkles: Attention Mechanism
- [x] **Bahdanau Attention**; *NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE* (ICLR 2015) [paper](https://arxiv.org/pdf/1409.0473.pdf)

- [x] **Multi-Head Attention**; *Attention Is All You Needs* (NIPS 2017) [paper](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

## :massage: Conversational AI

### Open-Domain
- Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index, ACL
- Kelvin Guu의 REALM, ACL
- [ ] **DPR**; *Dense Passage Retrieval for Open-Domain Question Answering* (20.04.10) [paper](https://arxiv.org/pdf/2004.04906.pdf)
  - [Huffon님 소개자료](https://www.facebook.com/111809756917564/posts/276190540479484/) 

## :art: Generative Model

### GAN
- [ ] **Original GAN**; *Generative Adversarial Net* (NIPS 2014) [paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

## :monkey_face: Meta Learning

## :brain: Reinforcement Learning
- [x] **Policy Gradient Theorem** *Policy Gradient Methods for Reinforcement Learning with Function Approximation* (NIPS 2000) [paper](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

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
  - with A3C

## :chart_with_upwards_trend: Financial Mathematics & Engineer

## :cat2: Theoretical Deep Learning

Batch Normalization

Lipschitz gradient

Global Batch Normalization

Input Covariate Shift

Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

How Does Batch Normalization Help Optimization?

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
