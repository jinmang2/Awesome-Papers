# Residual Connection
[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)에서 제시된 방법
[Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431)이 논문도 읽어라.

- 아래 수식이 Residual Connection을 나타냄.

$$\begin{array}c
y_l=h(x_l)+F(x_l,W_l)\\\\
x_{l+1}=f(y_l),\quad where\;h(x_l)=x_l\\
\end{array}$$
- identity mapping

- In general,
$$\begin{array}c
x_2=x_1+F(x_1,W_1)\\\\
x_3=x_2+F(x_2,W_2)=x_1+F(x_1,W_1)+F(x_2,W_2)\\\\
\vdots\\\\
x_L=x_l+\sum_{i=1}^{L-1}F(x_i,W_i)
\end{array}$$

- Differentiate, (use chain-rule)
$$\cfrac{\sigma\epsilon}{\sigma x_l}=
\cfrac{\sigma\epsilon}{\sigma x_L}\cdot\cfrac{\sigma x_L}{\sigma x_l}=
\cfrac{\sigma\epsilon}{\sigma x_L}
(1+\cfrac{\sigma}{\sigma x_l}\sum_{i=1}^{L-1}F(x_i,W_i))$$

- $\cfrac{\sigma\epsilon}{\sigma x_L}$은 상위 layer의 gradient값이 변하지 않고 하위 layer에 전달되는 것을 보여줌.
- 즉, layer를 거칠수록 gradient가 사라지는 vanishing gradient 문제를 완하시킴
- 또한 forward path나 backward path를 간단하게 표현할 수 있게 됨
- Highway Gateway는 Residual Connection의 상위 호환(ResNet의!)

출처:
- https://pozalabs.github.io/transformer/
- https://github.com/YBIGTA/DeepNLP-Study/wiki/Residual-Networks-Behave-Like-Ensembles-of-Relatively-Shallow-Networks
