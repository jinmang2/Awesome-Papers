# Neural Machine Translation of Rare Words with Subword Units
Byte-Pair-Encoding을 적용!

## Abstract
NMT 모델은 fixed vocab로 작동하지만 번역은 open-vocabulary 문제를 가짐
Out-of-Vocabulury 문제를 어떻게 해결할 것이냐
본 논문에서 `subword unit`들의 sequence로 encoding rare와 unknown words를 통해 open-vocabulary의 효용을 키우고자 함
인간의 직관: 다양한 단어 class는 단어보다 더 작은 유닛이 번역 가능할 것
예로,
- instance name(character copying or transliteration)
- compounds(compositional translation)
- cognate and loanwords(phonological and morphological transformation)

등에 더 유리할 것

본 논문에선 `n-gram`과 `byte-pair-encoding`을 비교할 것

`back-off dictionary`

## 1. introduction
드문 언어를 번역하는 것이 계속 문제였음
agglutination, compounding과 같은 형태의 언어들도 어려움
이를 번역하기위해서는 단어 그 이하의 단위가 필요
`variable-length representation`은 `fixed-length vector`보다 단어를 encoding하는데 훨씬 직관적임

단어단위의 nmt 모델의 항시 문제는 `back-off` dictionary look-up
아래 논문 참고
- (Jean et al., 2015; Luong et al., 2015b)

언어마다 형태학적 문법의 정도가 다양하기 때문에 1-1 correspondence가 항시 존재하진 않는다.
- 표현이 이뻐서 기록한다(나도 어쩔 수 없는 수학과이다.)
- `For instance, there is not always a 1-to-1 correspondence between source and target words because of variance in the degree of morphological synthesis between languages, like in our introductory compounding example`

단어 레벨 모델은 보지 못한 단어를 번역/생성하는 것이 불가능
`Copying unknown words into the target text` 방식이 2015년 제시됐다. 이는 이름 등에 대한 전략으로는 합리적이었지만 형태학적인 변화와 Transliteration(복구했을 시 원래의 문자 체계를 유지)은 특히 alphabet 단위로 다른 정보가 필요

자, `word-level`이 아닌 `subword units`로 nmt 모델을 기동시키는 연구를 진행!

우리 논문은 아래 두 contribution을 했다.
- `open-vocabulary neural machine translation`이 rare-word을 subword unit으로 encoding하여 가능케됨!
    - 더 많은 단어 사전 혹은 back-off dictionary를 사용하는 것보다 더 효과적이고 간단!
- `byte-pair-encoding`을 `word segmentation` 태스크에 알고리즘을 더 복잡하게 만들어 적용!
    - BPE는 `variable-length character sequences`의 고정된 크기의 vocab을 통해 open vocabulary 표현을 가능케하고 이는 nn모델의 매우 적합한 word segmentation 전략이 됨!

## 2. Neural Machine Translation
조경현 교수님 얘기 나오네, 진짜 대단하신분!
`Bahdanau attention`(리뷰 폴더의 attention 논문) 외에 더 특별한 부분을 가했다네

nmt system은 rnn을 활용한 encoder-decoder network으로 구현 (당시에는 그랬음)

encoder는 biLM with GRU
decoder는 uniLM with GRU지만 이전 단어, 이전 히든 상태와 context vector $c_i$까지하여 업데이트 된다. $c_i$는 annotation $h_j$의 가중합으로 계산되며 annotation $h_j$는 _alignment model_ $\alpha_{ij}$로 계산된다. _alignment model_ 은 단일 feedforward neural network, 역전파를 통해 network의 param을 동시에 학습

이 논문보니 SGD로 병렬 corpus로 학습을 수행했다더라.
아주 작은 `beam`으로 번역(inference)을 수행

## 3. Subword Translation
이 논문의 주된 동기는 몇몇 단어의 번역이 `transparent`하냐!!!
- 이걸 `투명하냐`로 번역하려한 나놈 반성해라... 이래가지고 언제 실력 늘을래!
- `in that`: ~라는 점에서! 종속 접속사
- 이는 `competent(유능한)` 번역기에 의해 번역가능하다는 것을 의미한단다
- **형태소와 음소와 같은 알고있는 subword unit들의 번역에 기반하여 한 번역이 가능할 것인가?**

Note that:
- Morpheme: 형태소
- Phoneme: 음소

뭐, 독일어, 영어 등에서 1.named entities 2.cognate and loanwords 3. morphologically complex words

오... 모르는 단어들을 공부해보자.
- `named entities`: 개체명
- `transcription` or `transliteration`
    - `transcription`(전사): 언어학에서 언어의 음성을 일정한 규칙에 근거해 문자 표기하는 것
        - 언어의 말소리를 특정 문자로 표기하는 것
        - [음성->문자]의 과정을 가리킴
        - 국어의 로마자 표기법(2002)에 따르면, 전사법은 읽기 편하게 하는 것에 중점을 둠
        - 때문에 글과 발은이 다르면 이를 반영
        - 웃옷[우돋] -> odot
    - `transliteration`(전자): 한 문자 체계를 다른 문자 체계로 바꾸어 쓰는 것
        - 전자한 후에 원래의 문자 체계로 복구할 수 있어야 한다
        - [문자->문자]의 과정을 가리킴
        - 국어의 로마자 표기법(2002)에 따르면, 전자법은 학술 논문 등에 쓰기 위해 원래 한글을 오류 없이 복원할 수 있는 표기를 목적으로 하고 있기 때문에 글자 그대로 표기
        - 웃옷 -> us-os
- `syllabary`: 음절문자
    - 한 음절의 한 글자로 나타내는 문자
    - 표의 문자에서 발달할 것이 많음
        - `ideogram`: 표의문자, 뜻글자
    - 일본의 가나가 대표
- `cognates`: 동계어
    - 어느 한 단어가 다른 한 언어와 일치하는 기원이 있는 단어
    - 일상적으로는 단순히 동의어 관계를 가지고 있다는 의미로 활용
    - 다른 의미를 지닌 언어적인 변화를 거친 동계어는 발음상으로 완전 다를 수 있다
    - cow와 beef는 둘다 똑같이 인도-유럽어족적인 루트 ``*gʷou-`` 를 가짐.
    - 이 경우, cow는 독일어족을 통해 발달되었지만
    - beef는 이탈리아어파와 로망스어군을 거쳐 영어에 편입
- `loanwords`: 외래어
- `affixation`: 접사, affix를 생각해라! suffix도 있네.

---
Note that: 생물학에서의 `transcription`
- [전사](https://ko.wikipedia.org/wiki/%EC%A0%84%EC%82%AC_(%EC%83%9D%EB%AC%BC%ED%95%99))는 ``전사``로 생물학에서 DNA에 적혀 있는 유전정보를 mRNA로 옮기는 과정이라네
- RNA 중합효소가 이 과정을 맡는데
- 전사 과정에서 한쪽 가닥만을 정보로 삼아 옮겨적고
- RNA가 합성된 이후 DNA는 원상복구
- 전사는 어떻게 진행되는가?
- **개시(initiation)**
    - DNA는 평상시에 히스톤이라는 단백질과 함께 촘촘한 구조를 이룸
    - 때문에 전사 인자와 RNA 중합요소가 DNA에 접근할 수 없다
    - 전사가 일어나기 위해 우선 히스톤에 아세틸기(아세트산 CH3COOH에서 하이드록시기 - OH를 떼어낸 것)가 붙어 히스톤의 모양이 바뀌고 DNA의 구조가 풀림
    - 전사가 일어날 위치에 여러 단백질이 붙어 두 가닥이 서로 붙어 있던 DNA가 서서히 풀림
    - 이 떄 자발적으로 한 가닥이 두 가닥으로 되는 과정을 억제하기 위해(`왜 생길지는 내 고민의 단계가 아님, 프로세르를 파악`) 한 가닥으로 되어 있는 DNA에 특정 단백질(SSB, single-strand binding protein)이 붙어서 계속 한 가닥으로 존재하도록 함
- **신장(elongation)**, 변형이라고도 번역
    - 두 가닥의 DNA 사슬 중 한 가닥을 주형으로 하여 이에 상보적인 염기를 가진 RNA 뉴클레오타이드가 차례로 하나씩 결합하여 RNA를 합성
    - 이 과정에서 RNA 중합효소가 관여
    - 염기 T(티민) 대신 U(유라실)이 A(아데닌)과 결합한다는 것을 제외, DNA 복제와 같은 원리로 상보적인 염기를 가진 뉴클레오타이드가 결합
    - 주형 DNA 사슬의 뉴클레오타이드의 염기가 A(아데닌)이면 U(유라실)을 갖는 뉴클레오타이드가 C(사이토신)이면 G(구아닌)을 갖는 뉴클레오타이드가 와서 차례로 결합
    - 새로운 RNA 뉴클레오타이드가 들어오면 RNA의 염기와 주형DNA 사슬의 염기가 일시적으로 결합하여 염기쌍을 이루지만, 전사가 끝난 부분의 DNA는 다시 꼬여서 원래의 DNA 2중 나선을 형성
    - 전사가 끝난 RNA는 DNA주형가닥으로부터 분리
- **종결(termination)**
    - RNA를 합성하면서 주형DNA사슬을 따라 이동하던 RNA중합효소가 DNA주형 사슬 내의 특별한 염기 부분인 종결 신호에 도달,
    - 더 이상 RNA를 합성하지 못하고 DNA로부터 떨어져 나온다.
    - 이에 함께 새로 만들어진 RNA가닥도 떨어져 나옴
- 전사의 장점?
    - RNA를 중간 물질로 활용, 유전자 발현을 용이하게 조절 가능
    - 하나의 유전자로부터 여러 가닥의 RNA를 생성, 단기간에 충분한 양의 단백질을 생성 가능
    - 유전 정보를 담고 있는 DNA가 직접 단백질을 생성하면 반응성이 큰 아미노산 등에 의해 유전 정보가 손상될 수 있는데
    - 이를 방지하여 유전전보와 안전성을 획득할 수 있음

---

잡설이 길어졌는데, 가정하기로 드문 언어를 적합한 subword unit으로 분해하는 것은 nn이
