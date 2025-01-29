import newAlgo as na
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from newAlgo import (
    tit_for_tat, grim, tit_for_2_tats, random_choice, switch,
    pavlov, always_defect, always_cooperate, smart_player, agent_player
)

def analyze_strategies(trials=10, games_per_matchup=5):
    """
    Comprehensive analysis of prisoner's dilemma strategies.
    
    Args:
        trials: Number of rounds per game
        games_per_matchup: Number of games to play for each strategy pair
    """
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
    
    # Initialize results storage
    results = {name: 0 for name in strategies}
    average_scores = {name: 0 for name in strategies}
    total_games = {name: 0 for name in strategies}
    payoff_matrix = np.zeros((len(strategies), len(strategies)))
    cooperation_rates = {name: [] for name in strategies}
    
    # Track detailed matchup history
    matchup_history = []
    
    for i, (strategy_name, strategy_func) in enumerate(strategies.items()):
        for j, (opponent_name, opponent_func) in enumerate(strategies.items()):
            if strategy_name == opponent_name:
                continue
                
            strategy_total_score = 0
            strategy_cooperation_count = 0
            
            for game_num in range(games_per_matchup):
                player1 = na.Player(strategy_func)
                player2 = na.Player(opponent_func)
                game = na.Game(trials)
                score1, score2 = game.play(player1, player2)
                
                strategy_total_score += score1
                strategy_cooperation_count += sum(1 for move in player1.history if move == 0)
                
                # Record detailed matchup data
                matchup_history.append({
                    'Strategy': strategy_name,
                    'Opponent': opponent_name,
                    'Game': game_num + 1,
                    'Score': score1,
                    'Opponent_Score': score2,
                    'Cooperation_Rate': sum(1 for move in player1.history if move == 0) / len(player1.history)
                })
            
            avg_score = strategy_total_score / games_per_matchup
            payoff_matrix[i, j] = avg_score
            
            if avg_score > (trials * 2):  # Threshold for winning
                results[strategy_name] += 1
            
            average_scores[strategy_name] += avg_score
            total_games[strategy_name] += 1
            cooperation_rates[strategy_name].append(
                strategy_cooperation_count / (trials * games_per_matchup)
            )
    
    # Calculate final averages
    for strategy in strategies:
        average_scores[strategy] /= total_games[strategy]
        cooperation_rates[strategy] = np.mean(cooperation_rates[strategy])
    
    return {
        'wins': results,
        'average_scores': average_scores,
        'cooperation_rates': cooperation_rates,
        'payoff_matrix': payoff_matrix,
        'strategy_names': list(strategies.keys()),
        'matchup_history': pd.DataFrame(matchup_history)
    }