import random

# Payoff matrix constants
R = 3  # Reward for mutual cooperation
T = 5  # Temptation (defect when other cooperates)
S = 0  # Sucker's payoff (cooperate when other defects)
P = 1  # Punishment for mutual defection

# Define strategies
def always_cooperate(own_history, opponent_history):
    return 'C'

def always_defect(own_history, opponent_history):
    return 'D'

def tit_for_tat(own_history, opponent_history):
    if not opponent_history:
        return 'C'
    else:
        return opponent_history[-1]

def tit_for_two_tats(own_history, opponent_history):
    if len(opponent_history) >= 2 and opponent_history[-1] == 'D' and opponent_history[-2] == 'D':
        return 'D'
    else:
        return 'C'

def grudger(own_history, opponent_history):
    if 'D' in opponent_history:
        return 'D'
    else:
        return 'C'

def random_strategy(own_history, opponent_history):
    return random.choice(['C', 'D'])

def pavlov(own_history, opponent_history):
    if not own_history:
        return 'C'
    last_own = own_history[-1]
    last_opponent = opponent_history[-1] if opponent_history else 'C'
    if (last_own == last_opponent):
        return last_own
    else:
        return 'D' if last_own == 'C' else 'C'

# List of strategies
strategies = [
    ('Always Cooperate', always_cooperate),
    ('Always Defect', always_defect),
    ('Tit for Tat', tit_for_tat),
    ('Tit for Two Tats', tit_for_two_tats),
    ('Grudger', grudger),
    ('Random', random_strategy),
    ('Pavlov', pavlov)
]

def simulate_match(strategy1, strategy2, num_rounds):
    history_p1 = []
    history_p2 = []
    score_p1 = 0
    score_p2 = 0

    for _ in range(num_rounds):
        # Get actions
        action_p1 = strategy1(history_p1.copy(), history_p2.copy())
        action_p2 = strategy2(history_p2.copy(), history_p1.copy())

        # Calculate scores
        if action_p1 == 'C' and action_p2 == 'C':
            score_p1 += R
            score_p2 += R
        elif action_p1 == 'C' and action_p2 == 'D':
            score_p1 += S
            score_p2 += T
        elif action_p1 == 'D' and action_p2 == 'C':
            score_p1 += T
            score_p2 += S
        else:
            score_p1 += P
            score_p2 += P

        history_p1.append(action_p1)
        history_p2.append(action_p2)

    return score_p1, score_p2

def main():
    num_rounds = 200
    results = {name: 0 for name, _ in strategies}

    # Simulate all pairs
    for name_a, func_a in strategies:
        for name_b, func_b in strategies:
            score_a, score_b = simulate_match(func_a, func_b, num_rounds)
            results[name_a] += score_a
            results[name_b] += score_b

    # Display results
    print("Total points for each strategy across all matches:")
    for name in sorted(results, key=lambda x: results[x], reverse=True):
        print(f"{name}: {results[name]} points")

if __name__ == "__main__":
    main()