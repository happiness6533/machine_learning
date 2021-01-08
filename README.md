## 머신러닝 관점 정리중...
1. 풀이 방법: 정보의 추상화
    - 확률대수관점 MLE
    - 기하관점 벡터공간 변환
      
2. 지도 학습: 데이터 분류 trained by x and y > x를 넣으면 확률 y를 알려주는 확률분포를 얻는다
    - 고전: svm hmm naive bayes regression
    - 현대: dnn cnn(비시계열 데이터) rnn trans bert gpt(시계열 데이터)
    - 응용: yolo resnet style-transfer concentrate

3. 비지도학습: 데이터 분류 trained by x > x를 넣으면 확률 y를 알려주는 확률분포를 얻는다
    - 이진분류: gaussian abnormaly > svm: 1개의 확률분포를 얻는다
    - 다중분류: k-means > gm > em > vi: 여러개의 확률분포를 얻는다
    
4. 지도 + 비지도: 데이터 생성 by latent z
    - find latent: pca kpca t-sne encoder > x를 넣으면 latent z를 알려주는 확률분포를 얻는다
    - create data: decoder > 데이터를 높은 확률로 샘플링하는 확률분포를 얻는다
    - 차원축소 이진분류 단일생성: encoder + decoder = ae
    - 차원축소 다중분류 + 다중생성: vae > gan

5. 강화학습: 셀프 판단 or 행동
    - 정책기반: 폴리시이터 q러닝 reinforce
    - 가치기반: 밸류이터 sarsa
    - 가치정책기반: deepsarsa dqn a2c

6. 텐서플로우와 케라스를 활용해서 핵심 알고리즘을 구현하고 현실의 문제를 해결

## 강화학습 관점 정리중...
1. 주어진 문제를 mdp로 구성
    - mdp 핵심 구성요소
        - 상태: 유한 / 무한
        - 행동: 상태를 변수로 가지고 행동을 결과로 가지는 확률분포함수를 정책이라고 한다
        - 보상

    - mdp 하이퍼 파라미터
        - 상태변환확률 : 행동에 의해 어떤 상태에서 다른 상태로 변화할 확률들의 집합
        - 감가율

    - mdp 풀이 = 현재 정책에 따른 real / semi-real v(s) 얻기 + 얻은 v(s)로 정책 업데이트 >> 반복 = optimal v(s) + optimal policy(s) 획득
        - 어떤 타임 t에서의 어떤 상태 s에서, 현재 정책에 따른, 감가율을 고려한, 미래에 얻을 수 있는 모든 보상의 합 = gt
        - 어떤 타임 t에서의 어떤 상태 s에서, 현재 정책에 따른, 감가율을 고려한, 미래에 얻을 수 있는 모든 보상의 합의 기대값 = v(s)

2. mdp를 풀이하기 위한 value and policy 업데이트 방법
    - v(s) 업데이트 방법1
        - 모든 상태 s가 유한한 경우, 모든 v(s)를 직접 계산할 수 있다
        - gt의 정의로부터 아래의 식을 유도한다
        - v(s) = 시그마(행동 확률 * (reward + 감가율 * v(s'))
        - 위의 수식에서, 현재 상태 s와 어떤 행동 a에 대해 얻어지는 상태 s'의 관계는
        - s에서 a를 해서 s'이 됨
        - 과 같으므로, 우리는 v(s')을 s와 a에 대한 함수로도 생각할 수 있다. 이러한 함수를 q 함수라고 한다면 다음이 성립한다
        - v(s') = q(s, a)
        - 따라서 상태 s는 가능한 행동 a들에 대해 여러 q(s, a)를 가진다
        - 따라서 위의 식을 다시 쓰면 아래와 같다
        - v(s) = 시그마(행동 확률 * (reward + 감가율 * q(s, a))
        - 여기까지 따라왔다면, 위의 수식을 조금 더 일반화시키면 아래와 같음을 이해할 수 있다
        - q(s, a) = 시그마(행동 확률 * (reward + 감가율 * q(s', a'))
        - 결론적으로 아래의 두 수식을 활용한다
        - v(s) = 시그마(행동 확률 * (reward + 감가율 * v(s')): 정책 이터레이션, 가치 이터레이션
        - q(s, a) = 시그마(행동 확률 * (reward + 감가율 * q(s', a')): sarsa, q러닝, deepSarsa, dqn

    - v(s) 업데이트 방법2
        - 모든 상태 s가 무한한 경우: 모든 v(s)를 직접 계산할 수 없다: mc(몬테카를로) 방법을 사용한다
        - 에피소드1: 지나온 경로를 기록하고, 경로에 따른 gt를 각 스테이트에 기록하자
        - 에피소드2: 지나온 경로를 기록하고, 경로에 따른 gt를 각 스테이트에 기록하자
        - 에피소드 반복
        - 이제 기록된 모든 스테이트에 따른 gt 기록들의 평균을 구하자 = v(s)
        - 이 평균은 지금 정책에 따라 얻어지는 참 가치함수에 수렴한다
        - 그런데 이렇게 하려면, 에피소드 1000번동안 있었던 모든 걸 다 기록해야 되는데, 이건 메모리가 많이 든다
        - 따라서 테크닉을 사용, 임의의 n번째 에피소드가 종료되면 즉시 v(s)를 업데이트 해도
        - 1000번의 에피소드를 마치고 v(s)를 업데이트한 결과와 같도록 수식을 구성한다
        - 업데이트할 v(s) = 기존의 v(s) + 1/n * (현재 gt - 기존의 v(s))

    - v(s) 업데이트 방법3
        - 모든 상태 s가 무한한 경우 : 모든 v(s)를 직접 계산할 수 없다 : td(temporal difference) 방법을 사용한다
        - full 에피소드 없이, 액션에 일어날 때마다 v(s) 실시간 업데이트
        - 업데이트할 v(s) = 기존의 v(s) + learning_rate * (현재 받는 리워드[에피소드 필요ㄴㄴ] + r * v(s')[에피소드 필요ㄴㄴ] - 기존의 v(s))
        - 반복해서 현재 정책에 따른 참 가치함수값을 얻고, 정책을 업데이트하고, 이를 반복한다

    - v(s)와 정책 업데이트 방법1 - 1
        - 모든 상태의 v(s)를 적당한 값으로 초기화하고, 반복 업데이트: 모든 상태의 현재 정책에 따른 real v(s)를 얻는다
        - 얻어진 real v(s)에서 그리디를 적용해서 정책 1회 업데이트
        - 위의 2과정을 반복

    - v(s)와 정책 업데이트 방법1 - 2
        - 모든 상태의 v(s)를 적당한 값으로 초기화하고, 1회 업데이트: 모든 상태의 현재 정책에 따른 semi-real v(s)를 얻는다
        - 얻어진 real v(s)에서 그리디를 적용해서 정책 1회 업데이트
        - 위의 2과정을 반복

    - v(s)와 정책 업데이트 방법2
        - 모든 상태의 v(s)를 적당한 값으로 초기화하고, 반복 업데이트: 모든 상태의 현재 정책에 따른 real v(s)를 얻는다
        - 위의 업데이트에서 max v(s')을 사용하면 그리디를 적용하는 효과가 있기 때문에, 정책을 따로 업데이트 할 필요가 없다
        - 위의 과정을 반복

3. mdp 문제 해결
    - 정책 이터레이션
        - 모든 상태 s가 유한한 경우: 정책 이터레이션
        - 현재 정책에 따라 얻어지는 참 가치함수값을 구하기 위해서 v(s) 업데이트를 여러번 반복한다
        - 이렇게 반복해서 얻어진 참 가치함수를 통해 정책을 업데이트 한다
        - 업데이트 된 새로운 정책에 따라 얻어지는 참 가치함수값을 구하기 위해서 v(s) 업데이트를 여러번 반복한다
        - 이렇게 반복해서 얻어진 참 가치함수를 통해 정책을 업데이트 한다
        - 이 과정을 반복하면, optimal 정책망이 얻어진다
        - 정책망에 따른 대강의 가치함수 >> 대강의 가치함수에 의한 정책망 업데이트 를 반복해도 같은 결론으로 수렴하게 된다

    - 가치 이터레이션
        - 모든 상태 s가 유한한 경우: 가치 이터레이션
        - 가치함수를 업데이트 할 때, 평균이 아니라, 최대값을 골라서 업데이트 한다
        - 무한히 반복하면 가치함수 자체가 옵티멀 정책을 내제하게 된다

    - sarsa
        - 상태 s가 너무 많은 경우: sarsa
        - td + 가치 이터레이션
        - 업데이트할 q(s, a) = 기존의 q(s, a)
        - + learning_rate * (현재 받는 리워드[에피소드 필요ㄴㄴ] + 감가율 * max q(s', a')[에피소드 필요ㄴㄴ]- 기존의 q(s, a))
        - 다음 스테이트 s'에서 최고의 큐를 반환해주는 a'을 그리디하게 선택해서 업데이트를 먼저 하고
        - 실제 이동은 입실론 탐험이 반영되지 못한, 이미 선택된 a'에 의해 이동한다
        - 맥스를 선택하기 떄문에, 정책평가가 이미 가치함수에 반영된다
        - 정책평가 없이 반복적인 가치함수 업데이트!

    - q러닝
        - 상태 s가 너무 많은 경우: q러닝
        - 큐함수를 이용해서 다음 큐함수를 업데이트 하면, 뒤의 액션에 의해 앞의 액션이 영향을 받아 갇히는 현상이 발생할 수 있다
        - 따라서, 정책은 explore + 그리디를 조합해서 쓰는건 똑같은데
        - 큐함수를 업데이트할 때, 다음 큐함수로부터 업데이트 하지 않고, 다음 state에서 얻어지는 max q(s, a)만을 사용해서 업데이트 하자
        - 다음 스테이트 s'에서 최고의 큐를 반환해주는 a'을 그리디하게 선택해서 업데이트를 먼저 하고
        - 실제 이동은 입실론 탐험이 반영된 택한 경로에 의해 이동한다
        - 맥스를 선택하기 떄문에, 정책평가가 이미 가치함수에 반영된다
        - 정책평가 없이 반복적인 가치함수 업데이트!

    - reinforce : 정책을 gt에 의해 선택한다
        - 현재 정책 = 현재 네트워크
        - 정책 업데이트 = 네트워크 업데이트(by gt)
        - 입실론을 쓰지 않는다 : 모든 행동이 확률적으로 나오기 때문에, 확률뷴포에 따르는 선택을 한다

    - deepSarsa
        - 살사 알고리즘 : 큐함수 업데이트 + 정책 업데이트
        - 현재 큐함수 = 네트워크가 근사하는 함수
        - 큐함수 업데이트 = 네트워크 업데이트
        - td : 다음 행동을 미리 그리디하게 선택하고 업데이트

    - dqn
        - q러닝 알고리즘 : 큐함수 업데이트 + 정책 업데이트
        - 현재 큐함수 = 네트워크가 근사하는 함수
        - 큐함수 업데이트 = 네트워크 업데이트
        - td : sars'을 리플레이 메모리에서 랜덤하게 추출, 다음 q를 maxq 선택 후 업데이트
        - 타겟 네트워크와 업데이트 네트워크를 분리한다

    - a2c = reinforce + deepsarsa
        - value/ q : 네트워크
        - action : 네트워크
        - 큐 신경망 업데이트 = 다음 큐함수 - 밸류(td) 제곱을 최소화하는 방향으로 업데이트
        - 정책 신경망 업데이트( : 큐함수 업데이트를 가져다가 쓴다
        - 입실론을 쓰지 않는다 : 모든 행동이 확률적으로 나오기 때문에, 확률뷴포에 따르는 선택을 한다

    - a3c = reinforce + dqn
        - 여러 에이전트가 동시에 작동해서 리플레이 메모리 문제를 해결하고 글로벌 네트워크를 업데이트 한다
        - multi-step으로 만들어진 q를 쓴다
        - q에 엔트로피 수식을 추가해서 exploration 경향성을 추가한다