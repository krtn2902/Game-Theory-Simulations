import newAlgo as na
import matplotlib.pyplot as plt

from newAlgo import (
    tit_for_tat,
    grim,
    tit_for_2_tats,
    random_choice,
    switch,
    pavlov,
    always_defect,
    always_cooperate,
    smart_player,
    agent_player
)

"""
Idea is to compare each algorithm against all other algorithms and cound the number of algorithms in which is wins.
Total Algorithms = n
Algorithm 1 scores a/n
Algorithm 2 scores b/n
and so on ...

A bar graph can will compare the results of each algorithm against all other algorithms.
"""

algorithms = [tit_for_tat, tit_for_2_tats, grim, random_choice, switch, agent_player, pavlov, always_cooperate, always_defect, smart_player]

# print(len(algorithms))

strategies = {
    "Tit for Tat": tit_for_tat,
    "Grim": grim,
    "Tit for 2 Tats": tit_for_2_tats,
    "Random Choice": random_choice,
    "Switch": switch,
    "Pavlov": pavlov,
    "RNN Agent": agent_player,
    "Always Defect": always_defect,
    "Always Cooperate": always_cooperate,
    "Smart Player": smart_player,
}

# Test each strategy against all others
results = {name: 0 for name in strategies}
trials = 10  # Number of iterations per game

for strategy_name, strategy_func in strategies.items():
    wins = 0
    print('-------------------------------------------')
    print(strategy_name)
    for opponent_name, opponent_func in strategies.items():
        player1 = na.Player(strategy_func)

        if strategy_name == opponent_name:
            continue
        
        print(opponent_name)
        player2 = na.Player(opponent_func)
        game = na.Game(trials)
        score1, score2 = game.play(player1, player2)
        print(score1, " ", score2)
        if score1 >= score2:
            wins += 1

    print(wins)
    results[strategy_name] = wins

print(results)

plt.bar(results.keys(), results.values())
plt.xlabel("Strategy")
plt.ylabel("Wins")
plt.title("Performance of Iterative Prisoner's Dilemma Strategies")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()