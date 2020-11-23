# A Structured Self-Attentive Sentence Embedding

## Abstract
본 논문은 `self-attention`이라 불리는 해석가능한 문장 임베딩을 추출하는 모델을 소개한다. vector를 사용하지 않고 embedding representation을 위해 2D matrix를 사용하며 행렬의 각 row vector가 문장의 다른 part에 집중한다. 본 연구에서 self-attention을 제안하며 모델의 특별한 `regularization` term을 제시한다. 부수적으로 embedding은 문장의 어떤 특정 부분이 embedding으로 부호화되는지 시각화하는 쉬운 방식으로 얻어진다. 우리는 `author profiling`, `sentiment classification`, `textual entailment`의 3가지 task에서 모델을 평가한다. 우리의 모델은 3가지의 task에서 다른 sentence embedding보다 더 높은 성능을 보였다.

## 1. Introduction
Bengio, Mikolov에 의해 소개된 `word embedding` 등의 각 단어의 의미적으로 유용한 분산 표현(`distributed representations`)을 학습하려는 많은 시도가 있었다. 그러나 아직 구(`phrase`)나 문장(`sentence`) 등에서 만족할만한 표현을 얻어야 하는 과제가 남아있다. 이러한 방법들은 일반적으로 두 범주로 떨어진다. 첫 째, Hill의 논문에서 제시한 `unsupervised learning`을 통해 universal한 sentence embedding을 학습시키는 것이다. 이에 대한 예시로는 아래의 논문들이 있다.
- SkipThought vectors (Kiros et al.,2015)
- Paragraph Vector (Le & Mikolov,2014)
- recursive auto-encoders (Socher et al.,2011;2013)
- Sequential Denoising Autoencoders (SDAE)
- FastSent (Hill et al., 2016)
- etc.

다른 범주는 모델이 특정 task를 특정하게 학습하는 것이다. 보통 downstream application과 결합되고 `supervised learning`을 통해 학습시킨다. 비록 `generic ones`이 대량의 unlabeled corpora에서 `semi-supervised learning`에서 사용할 수 있지만(많은 데이터를 활용할 수 있지만) 이보다 특정하게 sentence embedding을 학습시킨 것이 더 성능이 좋았다. 몇몇 모델들은 classification과 ranking (Yin & Schutze, 2015; Palangi et al.,2016; Tan et al.,2016; Feng et al.,2015)등의 다양한 task를 풀기 위해 sentence representation을 생성하는 중간 단계로 recurrent network (Hochreiter & Schmidhuber, 1997; Chung et al.,2014), recursive networks (Socher et al.,2013), convolutional networks (Kalchbrenner et al.,2014; dos Santos & Gatti, 2014; Kim, 2014)를 사용할 것을 제안한다. 이전 방법론의 제안들은 RNN의 final hidden state를 사용하거나 RNN들의 hidden states 혹은 convolved n-grams의 max-pooling을 사용하여 vector representation을 만드는 것을 포함한다. 추가적인 연구로 sentence representations (Ma et al.,2015; Mou et al.,2015b; Tai et al.,2015)을 향상시키기 위해 `parse`나 `dependence trees`같은 linguistic structures를 사용하기도 한다.

몇몇 task에서 사람들은 CNN 혹은 LSTM 모델의 최상단에 attention mechanism을 사용할 것을 제안한다. 이들은 sentence embedding(dos Santos et al.,2016)의 추가 source 정보를 추출할 guide로 attention mechanism을 사용할 것을 권하지만 감정 분석과 같은 task에서는 모델의 input으로 단일 문장이 들어오기 때문에 추출할 추가 정보가 없어 직접 적용할 수 없다. 이러한 경우, 가장 흔한 방법은 모든 time step마다 max-pooling혹은 averaging을 추가하거나(Lee & Dernoncourt,2016) 혹은 최신 time step의 hidden representation을 부호화된 embedding으로 채택하는 것이다(Margarit & Subramaniam, 2016).

**본 연구에서 가정하기로, recurrent model으로 모든 time step에서 의미적(semantic)정보를 추출하는 것은 상당히 어렵고 필요치 않다.** 자, 때문에 우리는 max-pooling 혹은 averaging step을 대체하기 위해 sequential models의 **`self-attention mechanism`** 을 제안코자 한다. `self-attention`은 문장의 다른 측면(정보)들을 multiple vector representation으로 추출한다. 이는 우리의 sentence embedding model의 LSTM 최 상위 layer에서 수행됬다. 이렇게 하면 추가적인 input없이도 어느 case에 집중할 것인지 알 수 있게된다! 게다가 이전 time step의 hidden representation에 직접적으로 접근하기 때문에 LSTM의 장기 메모리의 짐을 어느정도 덜어준다. `self-attentive sentence embedding`을 사용하여 얻어지는 부수적인 것으로 추출된 embedding은 아주 쉽고 명료하게 해석 가능하다는 것이다.

## 2. Approach

### 2.1 Model
제안할 sentence embedding model은 두 파트로 구성된다.
- bidirectional LSTM
- self-attention mechanism
    - LSTM hidden states의 weight vector들의 합들의 set을 제공

weight vector들의 합의 집합은 LSTM hidden states의 dot product로 계산되며 weighted LSTM hidden state의 결과를 문장의 embedding으로 고려한다. 예제에서 이는 downstream application에 적용하기 위해 multilayer perceptron과 결합되어 사용된다.

![self-attention_1](https://user-images.githubusercontent.com/37775784/77980113-728af800-7341-11ea-87f1-2a52d40affa5.PNG)

$\begin{array}l
\text{Figure 1. A sample model structure showing the sentence embedding model }\\
\text{combined with a fully connected and softmax layer for sentiment analysis (a).}\\
\text{The sentence embedding }M\text{ is computed as multiple weighted sums of hidden }\\
\text{states
    rom a bidirectional }LSTM(h1,\dots,h_n), \text{where the summation weights}\\
(A_{i1},\dots,A_{in})\text{ are computed in a way illustrated in (b). Blue colored shapes }\\
\text{stand for hidden representations, and red colored shapes stand for weights, }\\
\text{annotations, or input/output.}
\end{array}$

fully connected layer를 사용하며 Appendix A에 나와있듯이 maxtix sentence embedding의 2D 구조의 효용성을 위해 weight connections을 prune할 것을 제안한다.

$n$개의 token을 가지는 문장이 주어졌다고 가정하자. 문장은 word embedding의 sequence로 아래와 같이 표현된다.
$$S=(w_1,w_2,\dots,w_n)\cdots(1)$$
$w_i$는 문장의 $i$번째 단어의 $d$차원 word embedding의 vector standing이다. 즉, $w_i\in\mathbb{R}^d$
$S$는 2D maxtix로 모든 단어 임베딩을 연결한 sequence representation이다.
$S$의 shape는 $(n,\;d)$.

sequence $S$의 각 entry는 각각에 대해 독립적이다. 단일 문장 안의 인접한 단어들 사이의 연관성을 얻기 위해 연구에서는 bidirectional LSTM을 사용했다.
$$\begin{array}c
\overrightarrow{h_t}=\overrightarrow{LSTM}(w_t,\overrightarrow{h_{t-1}})\cdots(2)\\\\
\overleftarrow{h_t}=\overleftarrow{LSTM}(w_t,\overleftarrow{h_{t-1}})\cdots(3)
\end{array}$$

그 다음 hidden state $h_t$에서 얻어진 각 $\overrightarrow{h_t}$과 $\overleftarrow{h_t}$을 연결한다(concatenate). 각 unidirectional LSTM의 각 hidden unit number를 $u$라고 하자. 우리는 $n$개의 $h_t$를 $(n,\;2u)$의 크기를 가진 행렬 $H$로 아래와 같이 표기할 수 있다.
$$H=(h_1,h_2,\cdots,h_n)\cdots(4)$$

본 연구의 목적은 `variable-length sentence`를 `fixed size embedding`으로 부호화하는 것이다. 이는 $H$ 안의 $n$개의 LSTM hidden vectors의 `linear combination`으로 얻을 수 있다. **linear combination을 계산하는 것은 `self-attention mechanism`을 필요로 한다.** 그 연산은 아래와 같다.
$$a=softmax(w_{s2} tanh(W_{s1}H^T))\cdots(5)$$

$H:$ 전체 LSTM hidden states (input)
$a:$ weights vector (output)
$W_{s1}:$ $(d_a,\;2u)$ shape의 weight matrix
$w_{s2}:$ size $d_a$의 parameters vector
$d_a:$ 임의의 hyperparameter

$H$가 $(n,\;2u)$의 크기를 가졌기 때문에 `annotation vector` $a$는 $n$의 size를 가진다. $softmax()$ 함수는 가중함이 1이 되도록 보장한다. 그런 다음 $a$가 제공하는 가중치에 따라 LSTM 은닉 상태 $H$를 합산하여 입력 문장의 벡터 표현 $m$을 구한다.

이 vector representation은 보통 문장의 특정 구성 요소(단어나 구에 연결된 특별한 set과 같은)에 초점을 맞춘다. 때문에 우리는 이 표현이 문장의 의미적 요소를 반영하길 기대하지만, 전체 문장의 전반적인 의미적 요소는 문장의 다양한 부분에 깃들어 있다. 특히 긴 문장들이 그러하다(예를 들어, 두 절(clause)이 "and"로 연결돼있다면?). 때문에 문장의 전반적인 의미요소를 표현하기 위해서 우리는 문장의 서로 다른 부분들에 집중하는 다양한 $m$을 필요로 한다. 떄문에 우리는 여러가지 주의를 기울여야 한다. 문장으로부터 $r$개의 다른 부분이 추출되길 원한다고 하면 우리는 $w_{s2}$를 $(r,\;d_a)$로 확장시켜 $W_{s2}$로 표기하고 `annotation vector` $a$에서 확장된 `annotation matrix` $A$를 얻는다.
$$A=softmax(W_{s2} tanh(W_{s1}H^T))\cdots(6)$$

여기서 $softmax()$는 input의 2번째 차원에서 수행된다. 우리는 $Eq(6)$을 bias가 없는 hidden unit의 수는 $d_a$, parameter는 $\{W_{s2},W_{s1}\}$을 가진 2-layer MLP로 생각할 수 있다.

이제 embedding vector $m$은 $(r,\;2u)$ 차원의 embedding matrix $M$이 됐다. 우리는 `annotation matrix` $A$와 LSTM hidden states $H$의 $r$개의 가중합으로 이를 계산할 수 있고 수식으로는 아래와 같다.
$$M=AH\cdots(7)$$

### 2.2 Penalization Term
Embedding Matrix $M$이 $r$개의 다양한 hop을 추출하도록 페널티를 부과한다.
평가하기 최상의 방법은 2개의 summation weight vector 사이의 `Kullback Leibler divergence`를 정의하는 것이다. 그러나 우리의 case에서 아주 stable하지 않다는 것을 밝혀냈다. 짐작하기론, KL divergence를 maximize할 때 각기 다른 softmax output unit에 대해 0값이 굉장히 많은 annotation matrix $A$를 최적화해야하는데 이 많은 양의 0값이 학습 불균형을 만든다고 생각된다. 또한 우리는 각 row가 단편적인 의미 정보를 가지기를 원하고 softmax의 output 확률이 이를 반영하길 원하는데 KL은 우리가 원하는 정보를 제공하지 않는다. 그러나 KL penalty가 우리의 기대를 어느정도 충족시켜줄 수 있다.

앞서말한 단점을 극복할 새로운 penalization term을 소개한다. KL divergence penalization과 비교하여 새로운 규제는 1/3의 연산만을 필요로 한다. 행렬 $A$와 이의 transpose에 `dot product`를 실시하고 identity matrix를 빼고 이를 redundancy measure로 사용한다.
$$P={\Vert (AA^T-I) \Vert_F}^2\cdots(8)$$

$\Vert \cdot \Vert_F$은 행렬의 `Frobenius norm`이다. L2 regularization을 더한 것과 유사하게 새로운 penalization term $P$는 계수를 곱하고 downstream application에 관련된 original loss를 최소화할 때 같이 적용된다.

$A$의 서로 다른 summation vector $a^i$와 $a^j$를 생각해보자. softmax연산으로 행렬 $A$의 각 entry의 합은 1이다. 그러므로 이를 이산 확률 분포의 확률메져로 여실 수 있다. 행렬 $AA^T$의 diagonal element가 아닌 원소 $a_{ij}(i\neq j)$에 대해 이는 두 확률 분포의 elementwise product의 합으로 표현된다.
$$0<a_{ij}=\sum_{k=1}^{n}{a_k^i a_k^j}<1\cdots(9)$$

$a_k^i,\;a_k^j:$ $a^i$와 $a^j$의 $k$번째 원소
두 분포 $a^i$와 $a^j$가 서로 overlap되는 부분이 없는 극단적인 case를 고려하면, $a_{ij}$가 될 것이다. 바꿔말하면 겹치는 부분이 조금이라도 생기면 양수라는 얘기고 두 분포가 완전히 같고 오직 한 단어에만 집중한다면 이 최댓값은 1의 값을 가질 것이다. 위 식에서 $AA^T$에 단위 행렬 $I$를 빼서 $AA^T$의
