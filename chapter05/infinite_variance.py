#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# 비활성 정책의 최초 MC에 대한 결과

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACTION_BACK = 0
ACTION_END = 1

# behavior policy = epsilon-greedy policy
def behavior_policy():
    return np.random.binomial(1, 0.5)

# target policy = greedy policy
def target_policy():
    return ACTION_BACK

# one turn
def play():
    # track the action for importance ratio
    trajectory = []
    while True:
        action = behavior_policy()
        trajectory.append(action)
        if action == ACTION_END:
            return 0, trajectory
        # ACTION_BACK
        # np.random.binomial(n, p) : p 확률인 이항 분포를 n 번 실행했을 때의 결과. 성공 : 1, 실패 : 0.
        
        # p = 0.5 일때는 behavior policy와 동일하기 때문에, 추정값이 1로 잘 수렴한다.
        # p 를 바꿔가면서 분산을 확인해보자.
        # p 의 확률로 MDP 자기 자신에게 향한다.
        p = 0.9
        if np.random.binomial(1, p) == 0:
            return 1, trajectory

def figure_5_4():
    runs = 10
    episodes = 100000
    for run in range(runs):
        rewards = []
        for episode in range(0, episodes):
            reward, trajectory = play()
            # 마지막 action 이 action_end 일 떼,
            if trajectory[-1] == ACTION_END:
                rho = 0
            else:
                rho = 1.0 / pow(0.5, len(trajectory))
            rewards.append(rho * reward)
        rewards = np.add.accumulate(rewards)
        estimations = np.asarray(rewards) / np.arange(1, episodes + 1)
        plt.plot(estimations)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Ordinary Importance Sampling')
    plt.xscale('log')

    plt.savefig('sutton_rl/chapter05/figure_5_4.png')
    plt.close()

if __name__ == '__main__':
    figure_5_4()
