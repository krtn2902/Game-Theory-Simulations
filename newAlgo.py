import random

class Player:
    def __init__(self, strategy):
        self.strategy = strategy 
        self.history = []  

    def make_move(self, opponent_history):
        return self.strategy(self.history, opponent_history)

    def update_history(self, move):
        self.history.append(move)

class Game:
    def __init__(self, trials):
        self.game_length = trials
        self.player1_score = 0
        self.player2_score = 0

    def play(self, player1, player2):
        for trial in range(self.game_length):
            move_player1 = player1.make_move(player2.history)
            move_player2 = player2.make_move(player1.history)

            player1.update_history(move_player1)
            player2.update_history(move_player2)

            self.calculate_payoff(move_player1, move_player2)

        return self.player1_score, self.player2_score

    def calculate_payoff(self, move_player1, move_player2):

        # Payoff matrix
        # CC - (3, 3), DD - (0, 0) 
        # CD - (0, 5), DC - (5, 0),     

        if move_player1 == 0 and move_player2 == 0: # Both cooperate
            self.player1_score += 3
            self.player2_score += 3
        elif move_player1 == 0 and move_player2 == 1: # Player 1 cooperates, player 2 defects
            self.player1_score += 0
            self.player2_score += 5
        elif move_player1 == 1 and move_player2 == 0: # Player 1 defects, player 2 cooperates
            self.player1_score += 5
            self.player2_score += 0
        else:  # Both defect
            self.player1_score += 1
            self.player2_score += 1

# Strategy functions
def tit_for_tat(player_history, opponent_history): # defect if opponent defected in previous move
    if not opponent_history:
        return 0
    return opponent_history[-1]

def grim(player_history, opponent_history): # cooperate until the opponent defects once, then always defect
    if not opponent_history:
        return 0 
    if opponent_history[-1] == 1 or player_history[-1] == 1:
        return 1  # Defect forever
    return 0

def tit_for_2_tats(player_history, opponent_history):
    if len(opponent_history) < 2:
        return 0  
    if opponent_history[-1] == 1 and opponent_history[-2] == 1:
        return 1  
    return 0

def random_choice(player_history, opponent_history): # picks a random move
    return random.randint(0, 1)

def switch(player_history, opponent_history): # alternatively changes choice
    if not opponent_history:
        return 0
    if player_history[-1] == 1:
        return 0
    return 1

def pavlov(player_history, opponent_history):
    if not opponent_history:
        return 0
    if player_history[-1] == opponent_history[-1]:
        return 0
    return 1


def always_defect(player_history, opponent_history):
    return 1
def always_cooperate(player_history, opponent_history):
    return 0

def smart_player(player_history, opponent_history): 
    # a approach to be implemented here
    return 0


player1 = Player(strategy=tit_for_tat)  
player2 = Player(strategy=random_choice)  

game = Game(trials=10)
score_player1, score_player2 = game.play(player1, player2)

print(player1.history)
print(player2.history)

print(f"Player 1's score: {score_player1}")
print(f"Player 2's score: {score_player2}")

winner =  1
if score_player1 < score_player2:
    winner = 2


print(f"WINNER: player {winner}")