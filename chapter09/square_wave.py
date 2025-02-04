#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# wrapper class for an interval
# readability is more important than efficiency, so I won't use many tricks

# 엉성한 부호화 예제
# 상태 집합을 연속적인 2차원 공간에 나타날 때,
# 각 상태가 상태 집합(원으로 표현)에 포함되어 있으면 1, 없으면 0으로 binary feature 로 나타내며
# 상태 집합에 상태가 겹쳐진 특징을 이용하여 상태를 표현한느 것을 엉성한 부호화라고 한다.
# 선형 gradient descent 함수 근사를 가정하여 원의 개수와 크기의 효과에 대해서 알아보기 위한 예제
# 각 원(상태 집합)마다 파라미터 1개씩 가지고 있다.

# 해당 예제는 1상태 집합을 연속적인 1차원 공간에서 나타날 때의 예제
class Interval:
    # [@left, @right)
    # left와 right는 
    def __init__(self, left, right):
        self.left = left
        self.right = right

    # whether a point is in this interval
    # 아래 부등식이 성립하면 true, 성립하지 않으면 false return.
    def contain(self, x):
        return self.left <= x < self.right

    # length of this interval
    def size(self):
        return self.right - self.left

# domain of the square wave, [0, 2)
DOMAIN = Interval(0.0, 2.0)

# square wave function
def square_wave(x):
    if 0.5 < x < 1.5:
        return 1
    return 0

# get @n samples randomly from the square wave
def sample(n):
    samples = []
    for i in range(0, n):
        # class Interval에 left(0)와 right(2) 사이에서 uniform 하게 value sampling 
        x = np.random.uniform(DOMAIN.left, DOMAIN.right)
        y = square_wave(x)
        samples.append([x, y])
    return samples

# wrapper class for value function
class ValueFunction:()
    # @domain: domain of this function, an instance of Interval
    # @alpha: basic step size for one update
    def __init__(self, feature_width, domain=DOMAIN, alpha=0.2, num_of_features=50):
        
        # feature_width = [0.2 0.4 1.0], 이게 의미하는 바가 뭘까
        self.feature_width = feature_width
        # 50개의 특징 존재. 즉 50개의 구간이 존재한다.
        self.num_of_featrues = num_of_features
        self.features = []
        
        self.alpha = alpha
        self.domain = domain

        # there are many ways to place those feature windows,
        # following is just one possible way
        step = (domain.size() - feature_width) / (num_of_features - 1)
        left = domain.left
        for i in range(0, num_of_features - 1):
            self.features.append(Interval(left, left + feature_width))
            left += step
        self.features.append(Interval(left, domain.right))

        # initialize weight for each feature
        # 각 구간에 대한 weight
        self.weights = np.zeros(num_of_features)

    # for point @x, return the indices of corresponding feature windows
    # feature(50개로 나눈 구간)에 square_wave에 있으면 active_feature
    def get_active_features(self, x):
        active_features = []
        for i in range(0, len(self.features)):
            if self.features[i].contain(x):
                active_features.append(i)
        return active_features

    # estimate the value for point @x
    def value(self, x):
        active_features = self.get_active_features(x)
        # sum이 꼭 들어가야하나...?
        # active_features는 [0, 2, 4] 와 같이 list 형태가 될 수 있다.
        return np.sum(self.weights[active_features])

    # update weights given sample of point @x
    # @delta: y - x
    def update(self, delta, x):
        active_features = self.get_active_features(x)
        delta *= self.alpha / len(active_features)
        for index in active_features:
            self.weights[index] += delta

# train @value_function with a set of samples @samples
def approximate(samples, value_function):
    for x, y in samples:
        delta = y - value_function.value(x)
        value_function.update(delta, x)

# Figure 9.8
def figure_9_8():
    num_of_samples = [10, 40, 160, 640, 2560, 10240]
    feature_widths = [0.2, 0.4, 1.0]
    plt.figure(figsize=(30, 20))
    axis_x = np.arange(DOMAIN.left, DOMAIN.right, 0.02)
    for index, num_of_sample in enumerate(num_of_samples):
        print(num_of_sample, 'samples')
        samples = sample(num_of_sample)
        value_functions = [ValueFunction(feature_width) for feature_width in feature_widths]
        plt.subplot(2, 3, index + 1)
        plt.title('%d samples' % (num_of_sample))
        for value_function in value_functions:
            approximate(samples, value_function)
            values = [value_function.value(x) for x in axis_x]
            plt.plot(axis_x, values, label='feature width %.01f' % (value_function.feature_width))
        plt.legend()

    plt.savefig('sutton_rl/chapter09/figure_9_8.png')
    plt.close()

if __name__ == '__main__':
    figure_9_8()