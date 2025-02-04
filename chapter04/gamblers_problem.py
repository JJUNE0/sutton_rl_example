#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# goal
# 목표 금액
GOAL = 100

# all states, including state 0 and state 100
# 도박사가 보유할 수 있는 모든 경우의 수의 금액
STATES = np.arange(GOAL + 1)

# probability of head
# 동전 앞면이 나올 확률
HEAD_PROB = 0.4


def figure_4_3():
    # state value
    # state value 0으로 초기화하고. goal 일 때만 보상 1 획득.
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    sweeps_history = []

    # value iteration
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in STATES[1:GOAL]:
            # get possilbe actions for current state
            # 내기에 거는 돈은 0~min(s.100-s).
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            # \Sigma_a \pi(a|s) [r^a_s+\gamma\Sigma_{s'}P^a_{ss'}v(s')]
            # P^a_{ss'}={ HEAD_PROB, 1-HEAD_PROB )
            # v_{k+1}(s)=\max_a\Sigma_{s',r}p(s',r|s,a)[r+\gamma*v_k(s')]
            for action in actions:
                action_returns.append(
                    HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])
            # 각 State에서 가능한 action을 하여 얻을 수 있는 가치함수 중 가장 큰 값을 가치 함수로 저장. 
            new_value = np.max(action_returns)
            state_value[state] = new_value
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break

    # compute the optimal policy
    # state_value 는 value iteration을 통해 max 값이 계산되어 저장됨.
    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(
                HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])

        # round to resemble the figure in the book, see
        # https://github.com/ShangtongZhang/sutton_rl/issues/83
        # np.round(action_returns[1:], 5) : 1 부터 인덱스의 끝까지 소숫점 5자리에서 반올림
        # action_returns 중 최대가 되는 index +1 을 policy에 저장.
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('sutton_rl/chapter04/figure_4_3.png')
    plt.close()


if __name__ == '__main__':
    figure_4_3()
