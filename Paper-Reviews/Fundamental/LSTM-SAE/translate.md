Unsupervised Pre-training of a Deep LSTM-based Stacked Autoencoder for Multivariate Time Series Forecasting Problems
===

## LSTM-SAE
LSTM-based stacked autoencoder

## Reference
- [An innovative neural forecast of cumulative oil production from a petroleum reservoir employing higher-order neural networks (HONNs)](https://www.sciencedirect.com/science/article/abs/pii/S0920410513000582)

### Measure

$$RMSE=\sqrt{\cfrac{1}{n}\sum_{i=1}^{n}{(y_i^{obs}-y_i^{pred})}^2}$$

$$MAE=\cfrac{1}{n}\sum_{i=1}^{n}|y_i^{obs}-y_i^{pred}|$$

$$SMAPE=\cfrac{100}{n}\sum_{i=1}^{n}\cfrac{y_i^{obs}-y_i^{pred}}{y_i^{obs}+y_i^{pred}}$$

### Parameters selection
- Sequence length (lag)
- Batch size
- Number of units in the encoder layer
- Number of epochs for training the LSTM-SAE in the pre-training phase
- Number of epochs for training the pre-trained model in the fine-tuning phase
- Dropout rate
    - we use a dropout layer after each LSTM layer in the fine-tuninig phase

#### Hyperopt: Distributed Asynchronous Hyper-parameter Optimization library

### Data Preprocessing
데이터 전처리는 Time series forecasting of petroleum production using deep LSTM recurrent networks 논문의 내용대로 처리

- Step 1. Reduce noise from raw data
    - raw data를 smoothen시키고 noise를 감소시키기 위해 moving average filter를 사용
    - [An innovative neural forecast of cumulative oil production from a petroleum reservoir employing higher-order neural networks (HONNs)](https://www.sciencedirect.com/science/article/abs/pii/S0920410513000582)
    - 위 논문과 유사한 방법의 `low pass filter`로 사용
    - 특히 이 filter는 시계열의 매끄러운 추정치를 생성하기 위해 5개 point의 시간 범위 내에서 시계열 생산 데이터의 과거 데이터 지점의 weighted average를 제공
    - 이 step는 raw data와 관련된 가장 sharp한 step 응답을 유지함으로써 데이터의 무작위 노이즈를 줄이기 위해 필수적으로 통합된다.
- Step 2. Transform raw data to stationary data
    - 시계열은 특정한 trend를 가지는 non-stationary data
    - stationary data는 model이 이해하기 쉽고 더욱 skillfull한 예측 결과를 얻을 수 있음
    - 현 전처리 단계에서 data의 trend property를 제거 (increase vs decrease)
    - 나중에 예측 문제를 원래 척도로 되돌리고 비교 가능한 오차 점수를 계산하기 위해 추세를 다시 예측에 추가했다.
    - trend를 제거하는 표준 방법은 `Differencing of the data`
    - current observation - previous tim step's observation
- Step 3. Transform data into supervised learning
    - one-step ahead forecast를 사용 (다음 time step을 예측)
    - `lag time method`를 사용하여 시계열을 input x와 output y로 분리
    - 특별히 해당 논문에서는 lag1~lag6의 크기를 사용
- Step 4. Transform data into the problem scale
    - problem scale로 데이터 scale을 조절
    - LSTM의 default activation function은 `tanh(Hyperbolic tangent)` output in (-1, 1)
    - 예측 결과를 받은 후에는 scaled data를 original scale로 돌려줌
