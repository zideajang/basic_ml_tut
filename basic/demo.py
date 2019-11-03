# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
dist = stats.beta

def run_bayse_demo():
    num_iterations = 10**6
    # 将汽车随机放置一个门后面
    car_locations = np.random.randint(0,high=3,size=(num_iterations,))
    # 参赛选手随机选择一个门
    player_selection = np.random.randint(0,high=3,size=(num_iterations,))


    wins = (car_locations != player_selection)
    # wins = (car_locations == player_selection)
    win_pb = np.sum(wins) / num_iterations
    print(win_pb)

# 
def throw_a_coin(N):
    return np.random.choice(['H','T'],size=N,p=[0.5,0.5])



def run_game():
    N = 100
    throws = throw_a_coin(N)
    print("Throws: {}".format(throws))    
    print("Number of Heads: {}".format(np.sum(throws=='H')))
    print("Number of Heads/Total: {}".format(np.sum(throws=='H')/N))

def run_game_two():
    trials = [10,20,50,70,100,200,500,800,1000,2000,5000,7000,10000]
    plt.plot(trials,[np.sum(throw_a_coin(j)=='H')/np.float(j) for j in trials], 'o-',alpha=0.75)
    plt.xscale("log")
    plt.axhline(0.5,0,1,color='r')
    plt.xlabel('number of trials')
    plt.ylabel('probability of heads from simulation')
    plt.title('frenquentist probability of heads')
    plt.show()

def run_game_three():
    trials = [10,20,50,70,100,200,500,800,1000,2000,5000,7000,10000]

if __name__ == "__main__":
    run_game_two()
    # run_bayse_demo() #0.333497