# Label Smoothing
[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)에서 소개된 기법

- training동안 실제 정답인 label의 logit은 다른 logit보다 훨씬 큰 값을 가짐
- 이렇게 해서 model이 주어진 input $x$에 대한 label $y$를 맞추는 것
- 그러나 이렇게 되면 문제가 발생.
    - overfitting이 될 수도 있고 가장 큰 logit을 가지는 것과 나머지 사이의 차이를 점점 크게 만듦
    - 결국 model이 다른 data에 적응하는 능력을 감소시킴
- model이 덜 confident하게 만들기 위해, label distribution $q{(k|x)}=\delta_{k,y}$를 (k가 y일 경우 1, 나머지는 0) 다음과 같이 대체 가능

$$q^\prime(k|x)=(1-\epsilon)\delta_{k,y}+\epsilon u(k)$$

- 각각 label에 대한 분포 $u(k)$, smoothing parameter $\epsilon$.
- 위와 같다면, k=y인 경우 model은 $p(y|x)=1$이 아니라 $p(y|x)=(1-\epsilon)$이 됨
- 100%의 확신이 아닌 그보다 덜한 확신을 하게 되는 것
