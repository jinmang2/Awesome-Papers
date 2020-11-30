# Dropout
[Dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)에서 소개된 방법

![img](https://pozalabs.github.io/assets/images/dropout.PNG)

- neural network에서 unit들을 dropout시키는 것.
- 즉, 해당 unit을 network에서 일시적으로 제거
- 다른 unit과의 모든 connection이 사라짐
- 어떤 unit을 dropout할지는 random하게 정함
- dropout은 training data에 overfitting되는 문제를 어느정도 막아줌
- dropout된 unit들은 training되지 않는 것이니 training data에 값이 조정되지 않음
