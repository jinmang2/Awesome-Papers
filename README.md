# Awesome-Papers
:bulb: To be AI Researcher

## :office: NLP

### Word Vector Representation
- [x] *Distributed Representations of Words and Phrases and their Compositionality*

### Natural Langauge Understanding

## :sparkles: Attention Mechanism
- [x] **Multi-Head Attention**; *Attention Is All You Needs* (NIPS 2017) [link](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

## :art: Generative Model

### GAN
- [ ] **Original GAN**; *Generative Adversarial Net* (NIPS 2014) [link](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

## :brain: Reinforcement Learning
- [x] **Policy Gradient Theorem** *Policy Gradient Methods for Reinforcement Learning with Function Approximation* (NIPS 2000) [link](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

### :information_source: DeepMind
project: ShallowMinded

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
  
    exploits the multithreading capabilities of standard CPUs
    - [ ] **A3C (Asynchronosu Advantage Actor-Critic)**; *Asynchronous Methods for Deep Reinforcement Learning* (16.06.16, arxiv) [paper](https://arxiv.org/pdf/1602.01783.pdf)
      - [ ] **Intrinsic Motivation**; *Unifying Count-Based Exploration and Intrinsic Motivation* (16.12.07, arxiv) [paper](https://arxiv.org/pdf/1606.01868.pdf)
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

- [x] 20.09.18;	Decoupled Neural Interfaces Using Synthetic Gradients (16.08.29) [post](https://deepmind.com/blog/article/decoupled-neural-networks-using-synthetic-gradients)

- [x] 20.09.19;	Applying machine learning to radiotherapy planning for head & neck cancer (16.08.30) [post](https://deepmind.com/blog/announcements/applying-machine-learning-radiotherapy-planning-head-neck-cancer)

- [x] 20.09.21;	WaveNet: A generative model for raw audio (16.09.08) [post](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)

- [x] 20.09.22;	Putting patients at the heart of DeepMind Health (16.09.21) [post](https://deepmind.com/blog/announcements/putting-patients-heart-deepmind-health)

- [x] 20.09.23;	Announcing the Partnership on AI to Benefit People & Society (16.09.28) [post](https://deepmind.com/blog/announcements/announcing-partnership-ai-benefit-people-society)

- [x] 20.09.24;	Differentiable neural computers (16.10.12) [post](https://deepmind.com/blog/article/differentiable-neural-computers)

- [x] 20.09.25;	DeepMind and Blizzard to release StarCraft II as an AI research environment (16.12.04) [post](https://deepmind.com/blog/announcements/deepmind-and-blizzard-release-starcraft-ii-ai-research-environment)

- [ ] 20.09.26;	Reinforcement learning with unsupervised auxiliary tasks (16.12.17) [post](https://deepmind.com/blog/article/reinforcement-learning-unsupervised-auxiliary-tasks)

## :chart_with_upwards_trend: Financial Mathematics & Engineer

## :massage: Conversational AI

## :cat2: Theoretical Deep Learning

## :heart_eyes: Schmidhuber
