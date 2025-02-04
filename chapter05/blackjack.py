#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Nicky van Foreest(vanforeest@gmail.com)                        #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import sys
sys.path.append("/home/cocel/miniconda3/lib/python3.12/site-packages")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# tqdm은 **진행 상태(progress bar)**를 표시해주는 Python 라이브러리
from tqdm import tqdm

# actions: hit or stand
# 카드 추가
ACTION_HIT = 0
# 자신의 차례 건너뛰기
ACTION_STAND = 1  #  "strike" in the book
ACTIONS = [ACTION_HIT, ACTION_STAND]

# policy for player
# Policy_Player = [0~21]
# 게임 참여자의 카드 합이 12~19면 뽑고, 20,21 이면 뽑지 않는다. 
POLICY_PLAYER = np.zeros(22, dtype=np.int32)
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STAND
POLICY_PLAYER[21] = ACTION_STAND

# function form of target policy of player
def target_policy_player(usable_ace_player, player_sum, dealer_card):
    return POLICY_PLAYER[player_sum]

# function form of behavior policy of player
def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
    # 0.5 의 확률로 실행. (1 or 0.5) 중 택 1.
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT

# policy for dealer
POLICY_DEALER = np.zeros(22)
# 딜러는 보유한 카드의 합이 12~16 이면 뽑고, 17 이상이면 뽑지 않는다.
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND

# get a new card
def get_card():
    # 1~14 까지 랜덤으로 카드를 뽑고,
    # 뽑은 숫자와 10 중에 더 작은 값을 return 한다?? => 숫자 카드를 제외한 그림카드는 10으로 취급. 
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# get the value of a card (11 for ace).
def card_value(card_id):
    # 3항 연산자로, ACE 카드면 11, 그 외에는 카드 값 그대로 반환.
    return 11 if card_id == 1 else card_id

# play a game
# @policy_player: specify policy for player
# @initial_state: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# 사용가능한 에이스 카드가 있는지(0,1), 플레이어의 카드의 총 합(12~21), 딜러가 처음 낸 카드의 숫자(Ace~10) 총 200개의 state.
# @initial_action: the initial action
def play(policy_player, initial_state=None, initial_action=None):
    
    # player status
    usable_ace_player = 0

    # sum of player
    player_sum = 0

    # trajectory of player
    player_trajectory = []

    # whether player uses Ace as 11ed Robt을 제어하기 위한
    # dealer_card1 과 card2의 차이점은 무엇인가.
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False

    if initial_state is None:
        # generate a random initial state

        while player_sum < 12:
            # if sum of player is less than 12, always hit
            # 카드를 뽑고 (1~10)
            card = get_card()
            # 뽑은 카드에 ACE가 있는지 확인하고 더해준다.
            player_sum += card_value(card)

            # If the player's sum is larger than 21, he may hold one or two aces.
            if player_sum > 21:
                # assert 뒤의 조건이 참이 아닐 떼, Assert Error 발생.
                assert player_sum == 22
                # last card must be ace
                player_sum -= 10
            else:
                # card == 1 이면 usabla_ace_player = true가 된다.
                usable_ace_player |= (1 == card)

        # initialize cards of dealer, suppose dealer will show the first card he gets
        dealer_card1 = get_card()
        dealer_card2 = get_card()

    else:
        # use specified initial state
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    # initial state of the game
    state = [usable_ace_player, player_sum, dealer_card1]

    # initialize dealer's sum
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    
    # dealer_card1,2에 1이 있으면 True 
    # in 연산자는 tuple,list,dict 에 사용가능.
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)
    # if the dealer's sum is larger than 21, he must hold two aces.
    if dealer_sum > 21:
        assert dealer_sum == 22
        # use one Ace as 1 rather than 11
        dealer_sum -= 10
    assert dealer_sum <= 21
    assert player_sum <= 21

    # game starts!

    # player's turn
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            # get action based on current sum
            # policy_player 은 play 함수의 인자
            # action 은 hit or stand
            action = policy_player(usable_ace_player, player_sum, dealer_card1)

        # track player's trajectory for importance sampling
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])

        if action == ACTION_STAND:
            break
        # if hit, get new card
        card = get_card()
        # Keep track of the ace count. the usable_ace_player flag is insufficient alone as it cannot
        # distinguish between having one ace or two.
        ace_count = int(usable_ace_player)
        if card == 1:
            ace_count += 1
        player_sum += card_value(card)
        # If the player has a usable ace, use it as 1 to avoid busting and continue.
        while player_sum > 21 and ace_count:
            player_sum -= 10
            ace_count -= 1
        # player busts
        if player_sum > 21:
            # player 패배
            return state, -1, player_trajectory
        assert player_sum <= 21
        usable_ace_player = (ace_count == 1)

    # dealer's turn
    while True:
        # get action based on current sum
        action = POLICY_DEALER[dealer_sum]
        if action == ACTION_STAND:
            break
        # if hit, get a new card
        new_card = get_card()
        ace_count = int(usable_ace_dealer)
        if new_card == 1:
            ace_count += 1
        dealer_sum += card_value(new_card)
        # If the dealer has a usable ace, use it as 1 to avoid busting and continue.
        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1
        # dealer busts
        if dealer_sum > 21:
            # player 승리
            return state, 1, player_trajectory
        usable_ace_dealer = (ace_count == 1)

    # compare the sum between player and dealer
    assert player_sum <= 21 and dealer_sum <= 21
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory

# Monte Carlo Sample with On-Policy
def monte_carlo_on_policy(episodes):
    
    # player와 dealer가 가질 수 있는 state의 갯수는 200 개인데, 이를 ace 카드 사용 여부로 나눠서 배열을 만들었다.
    states_usable_ace = np.zeros((10, 10)) 
    states_no_usable_ace = np.zeros((10, 10))
    
    # initialze counts to 1 to avoid 0 being divided
    # 몇번의 샘플링이 이루어졌는지를 계산하여 평균 보상을 계산하는데 사용한다.
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))
    
    for i in tqdm(range(0, episodes)):
        # target_policy_player : 플레이어가 가지고 있는 카드 숫자의 총합에 따라서 action 을 선택하는 Policy
        # player_trajectory : [(usable_ace_player, player_sum, dealer_card1), action]
        _, reward, player_trajectory = play(target_policy_player)
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            player_sum -= 12
            dealer_card -= 1
            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usable_ace[player_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count

# Monte Carlo with Exploring Starts
def monte_carlo_es(episodes):
    # (playerSum, dealerCard, usableAce, action)
    state_action_values = np.zeros((10, 10, 2, 2))
    # initialze counts to 1 to avoid division by 0
    state_action_pair_count = np.ones((10, 10, 2, 2))

    # behavior policy is greedy : deterministic
    def behavior_policy(usable_ace, player_sum, dealer_card):
        usable_ace = int(usable_ace)
        player_sum -= 12
        dealer_card -= 1
        # get argmax of the average returns(s, a)
        # state_action_values[player_sum, dealer_card, usable_ace, :] 현재 상태에서 가능한 모든 action
        # 각 action 에 대한 평균 보상.
        values_ = state_action_values[player_sum, dealer_card, usable_ace, :] / \
                  state_action_pair_count[player_sum, dealer_card, usable_ace, :]
        print(values_)
        # 최댓값을 가진 행동들을 추출합니다. 즉, 여러 행동이 동일한 최대 보상을 가질 수 있기 때문에, 그 행동들만 선택
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # play for several episodes
    for episode in tqdm(range(episodes)):
        # for each episode, use a randomly initialized state and action
        initial_state = [bool(np.random.choice([0, 1])),
                       np.random.choice(range(12, 22)),
                       np.random.choice(range(1, 11))]
        initial_action = np.random.choice(ACTIONS)
        
        # episode 가 정수일 텐데 거짓일 수가 있나?
        current_policy = behavior_policy if episode else target_policy_player
        
        _, reward, trajectory = play(current_policy, initial_state, initial_action)
        
        first_visit_check = set()
        
        for (usable_ace, player_sum, dealer_card), action in trajectory:
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            state_action = (usable_ace, player_sum, dealer_card, action)
            if state_action in first_visit_check:
                continue
            first_visit_check.add(state_action)
            # update values of state-action pairs
            # 첫 방문 때만 update가 된다.
            state_action_values[player_sum, dealer_card, usable_ace, action] += reward
            state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1

    return state_action_values / state_action_pair_count

# Monte Carlo Sample with Off-Policy
def monte_carlo_off_policy(episodes):
    initial_state = [True, 13, 2]

    rhos = []
    returns = []

    for i in range(0, episodes):
        _, reward, player_trajectory = play(behavior_policy_player, initial_state=initial_state)

        # get the importance ratio
        numerator = 1.0
        denominator = 1.0
        for (usable_ace, player_sum, dealer_card), action in player_trajectory:
            if action == target_policy_player(usable_ace, player_sum, dealer_card):
                # behavior policy 의 확률이 1/2 이기 때문.
                denominator *= 0.5
            else:
                numerator = 0.0
                break
        # rho = (target policy)/(behavior policy)
        rho = numerator / denominator
        rhos.append(rho)
        returns.append(reward)

    # np.asarray() : 주어진 순차 데이터를 nparray로 변환.
    rhos = np.asarray(rhos)
    returns = np.asarray(returns)
    weighted_returns = rhos * returns

    # np.add.accumulate : 주어진 순차 데이터의 누적합.
    weighted_returns = np.add.accumulate(weighted_returns)
    rhos = np.add.accumulate(rhos)

    ordinary_sampling = weighted_returns / np.arange(1, episodes + 1)

    # np.errstate() :  나누기 연산에서 0으로 나누거나 NaN이 발생해도 경고를 출력하지 않고 무시
    # np.where(condition, a, b) : condition이 true일 때 a output. false 면 b output 
    with np.errstate(divide='ignore',invalid='ignore'):
        weighted_sampling = np.where(rhos != 0, weighted_returns / rhos, 0)

    return ordinary_sampling, weighted_sampling

def figure_5_1():
    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)

    states = [states_usable_ace_1,
              states_usable_ace_2,
              states_no_usable_ace_1,
              states_no_usable_ace_2]

    titles = ['Usable Ace, 10000 Episodes',
              'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes',
              'No Usable Ace, 500000 Episodes']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for state, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('sutton_rl/chapter05/figure_5_1.png')
    plt.close()

def figure_5_2():
    state_action_values = monte_carlo_es(500000)

    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)

    # get the optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)

    images = [action_usable_ace,
              state_value_usable_ace,
              action_no_usable_ace,
              state_value_no_usable_ace]

    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('sutton_rl/chapter05/figure_5_2.png')
    plt.close()

def figure_5_3():
    true_value = -0.27726
    episodes = 10000
    runs = 100
    error_ordinary = np.zeros(episodes)
    error_weighted = np.zeros(episodes)
    for i in tqdm(range(0, runs)):
        ordinary_sampling_, weighted_sampling_ = monte_carlo_off_policy(episodes)
        # get the squared error
        error_ordinary += np.power(ordinary_sampling_ - true_value, 2)
        error_weighted += np.power(weighted_sampling_ - true_value, 2)
    error_ordinary /= runs
    error_weighted /= runs

    plt.plot(np.arange(1, episodes + 1), error_ordinary, color='green', label='Ordinary Importance Sampling')
    plt.plot(np.arange(1, episodes + 1), error_weighted, color='red', label='Weighted Importance Sampling')
    plt.ylim(-0.1, 5)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel(f'Mean square error\n(average over {runs} runs)')
    plt.xscale('log')
    plt.legend()

    plt.savefig('sutton_rl/chapter05/figure_5_3.png')
    plt.close()


if __name__ == '__main__':
    figure_5_1()
    figure_5_2()
    figure_5_3()
