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
    opponent_payoff_matrix = np.zeros((len(strategies), len(strategies)))
    cooperation_rates = {name: [] for name in strategies}
    
    # Track detailed matchup history
    matchup_history = []
    
    for i, (strategy_name, strategy_func) in enumerate(strategies.items()):
        for j, (opponent_name, opponent_func) in enumerate(strategies.items()):
            if strategy_name == opponent_name:
                continue
                
            strategy_total_score = 0
            opponent_total_score = 0
            strategy_cooperation_count = 0
            
            for game_num in range(games_per_matchup):
                player1 = na.Player(strategy_func)
                player2 = na.Player(opponent_func)
                game = na.Game(trials)
                score1, score2 = game.play(player1, player2)
                
                strategy_total_score += score1
                opponent_total_score += score2
                strategy_cooperation_count += sum(1 for move in player1.history if move == 0)
                
                # Record detailed matchup data
                matchup_history.append({
                    'Strategy': strategy_name,
                    'Opponent': opponent_name,
                    'Game': game_num + 1,
                    'Score': score1,
                    'Opponent_Score': score2,
                    'Won': score1 > score2,
                    'Cooperation_Rate': sum(1 for move in player1.history if move == 0) / len(player1.history)
                })
            
            avg_score = strategy_total_score / games_per_matchup
            opponent_avg_score = opponent_total_score / games_per_matchup
            
            payoff_matrix[i, j] = avg_score
            opponent_payoff_matrix[i, j] = opponent_avg_score
            
            # New winning criteria: strategy must score higher than opponent
            if avg_score > opponent_avg_score:
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
        'opponent_payoff_matrix': opponent_payoff_matrix,
        'strategy_names': list(strategies.keys()),
        'matchup_history': pd.DataFrame(matchup_history)
    }

def plot_results(analysis_results):
    """Create comprehensive visualizations of the analysis results."""
    # 1. Overall Performance (Wins)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    bars = plt.bar(analysis_results['wins'].keys(), analysis_results['wins'].values())
    plt.xlabel("Strategy")
    plt.ylabel("Number of Wins Against Other Strategies")
    plt.title("Strategy Performance (Head-to-Head Wins)")
    plt.xticks(rotation=45, ha="right")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 2. Average Scores
    plt.subplot(122)
    bars = plt.bar(analysis_results['average_scores'].keys(), 
                   analysis_results['average_scores'].values())
    plt.xlabel("Strategy")
    plt.ylabel("Average Score")
    plt.title("Average Scores per Strategy")
    plt.xticks(rotation=45, ha="right")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Previous imports and functions remain the same until the plot_results function

    """Create comprehensive visualizations of the analysis results."""
    # Previous plots (1, 2) remain the same
    
    # 3. Win/Draw/Loss Matrix
    plt.figure(figsize=(10, 8))
    result_matrix = np.zeros_like(analysis_results['payoff_matrix'])
    
    # Create matrix with: 1 for win, 0 for draw, -1 for loss
    for i in range(len(analysis_results['strategy_names'])):
        for j in range(len(analysis_results['strategy_names'])):
            if i != j:
                strategy_score = analysis_results['payoff_matrix'][i, j]
                opponent_score = analysis_results['opponent_payoff_matrix'][i, j]
                if abs(strategy_score - opponent_score) < 0.01:  # Small threshold for floating-point comparison
                    result_matrix[i, j] = 0  # Draw
                elif strategy_score > opponent_score:
                    result_matrix[i, j] = 1  # Win
                else:
                    result_matrix[i, j] = -1  # Loss

    # Custom colormap: red for loss, yellow for draw, green for win
    colors = ['red', 'yellow', 'green']
    n_bins = 3  # 3 possible outcomes
    custom_cmap = plt.cm.RdYlGn
    
    plt.imshow(result_matrix, cmap=custom_cmap, vmin=-1, vmax=1)
    plt.colorbar(label='Outcome', ticks=[-1, 0, 1], 
                format=plt.FuncFormatter(lambda x, _: {-1: 'Loss', 0: 'Draw', 1: 'Win'}[x]))
    
    # Add text annotations
    for i in range(len(analysis_results['strategy_names'])):
        for j in range(len(analysis_results['strategy_names'])):
            if i != j:
                result = result_matrix[i, j]
                text = 'Draw' if result == 0 else ('Win' if result == 1 else 'Loss')
                color = 'black' if result == 0 else ('white' if result == -1 else 'black')
                plt.text(j, i, text, ha="center", va="center", color=color)
    
    plt.xticks(range(len(analysis_results['strategy_names'])), 
               analysis_results['strategy_names'], rotation=45, ha="right")
    plt.yticks(range(len(analysis_results['strategy_names'])), 
               analysis_results['strategy_names'])
    plt.title("Head-to-Head Outcomes (including Draws)")
    plt.xlabel("Opponent Strategy")
    plt.ylabel("Strategy")
    plt.tight_layout()
    plt.show()
    
    # Rest of the plots (4, 5) remain the same

    # Update the results dictionary to include draws
    strategy_results = {name: {'wins': 0, 'draws': 0, 'losses': 0} 
                       for name in analysis_results['strategy_names']}
    
    for i, strategy in enumerate(analysis_results['strategy_names']):
        for j, opponent in enumerate(analysis_results['strategy_names']):
            if i != j:
                if result_matrix[i, j] == 1:
                    strategy_results[strategy]['wins'] += 1
                elif result_matrix[i, j] == 0:
                    strategy_results[strategy]['draws'] += 1
                else:
                    strategy_results[strategy]['losses'] += 1
    
    # Print detailed summary statistics including draws
    print("\nStrategy Summary:")
    print("-" * 50)
    for strategy in analysis_results['strategy_names']:
        total_games = len(analysis_results['strategy_names']) - 1  # excluding self-play
        results = strategy_results[strategy]
        print(f"\n{strategy}:")
        print(f"Wins: {results['wins']} ({results['wins']/total_games:.1%})")
        print(f"Draws: {results['draws']} ({results['draws']/total_games:.1%})")
        print(f"Losses: {results['losses']} ({results['losses']/total_games:.1%})")
        print(f"Average Score: {analysis_results['average_scores'][strategy]:.2f}")
        print(f"Cooperation Rate: {analysis_results['cooperation_rates'][strategy]:.2%}")

# Rest of the code remains the same
    
    # 4. Cooperation Rates
    plt.figure(figsize=(10, 6))
    bars = plt.bar(analysis_results['cooperation_rates'].keys(), 
                   analysis_results['cooperation_rates'].values())
    plt.xlabel("Strategy")
    plt.ylabel("Cooperation Rate")
    plt.title("Average Cooperation Rate by Strategy")
    plt.xticks(rotation=45, ha="right")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Score Distribution Box Plot
    plt.figure(figsize=(12, 6))
    df = analysis_results['matchup_history']
    plt.boxplot([group['Score'].values for name, group in df.groupby('Strategy')],
                labels=analysis_results['strategy_names'])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Score Distribution by Strategy")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run analysis
    analysis_results = analyze_strategies(trials=100, games_per_matchup=5)
    
    # Generate visualizations
    plot_results(analysis_results)
    
    # Print detailed summary statistics
    print("\nStrategy Summary:")
    print("-" * 50)
    for strategy in analysis_results['strategy_names']:
        print(f"\n{strategy}:")
        print(f"Total Wins: {analysis_results['wins'][strategy]} out of {len(analysis_results['strategy_names'])-1} opponents")
        print(f"Win Rate: {analysis_results['wins'][strategy]/(len(analysis_results['strategy_names'])-1):.1%}")
        print(f"Average Score: {analysis_results['average_scores'][strategy]:.2f}")
        print(f"Cooperation Rate: {analysis_results['cooperation_rates'][strategy]:.2%}")