# Layer Normalization
[Layer Normalization](https://arxiv.org/abs/1607.06450)에서 제시된 방법

$$\begin{array}c
\mu^l=\cfrac{1}{H}\sum_{i=1}^{H}a_i^l\\\\
\sigma^l=\sqrt{\cfrac{1}{H}\sum_{i=1}^{H}{(a_i^l-\mu^l)}^2}
\end{array}$$

- 같은 layer에 있는 모든 hidden unit은 동일한 $\mu$와 $\sigma$를 공유
- 그리고 현재 input $x^t$, 이전의 hidden state $h^{t-1}$, $a_t=W_{hh}h^{t-1}+W_{xh}x^t$, parameter $g,b$가 있을 떄 다음과 같이 `normalization` 수행

$$h^t=f[\cfrac{g}{\sigma^t}\odot(a^t-\mu^t)+b]$$

- 이렇게 하면 gradient가 exploding하거나 vanishing하는 문제를 완화시키고 gradient 값이 안정적인 값을 가짐으로 더 빨리 학습시킬 수 있다.

출처: https://pozalabs.github.io/transformer/
