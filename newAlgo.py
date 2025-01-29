import random
import torch
import torch.nn as nn
import torch.optim as optim

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

    """
    Plays random move for first 5 trials, 
    After 5 trials,
        if opponent defected 80% times, always defect
        if cooperates 80% of the time, always defect
        else if tit for tat, always defect

    """
    if len(opponent_history) < 5:
        return random_choice(player_history, opponent_history)  # Random start
    
    # Analyze opponent behavior
    cooperate_count = opponent_history.count(0)
    defect_count = opponent_history.count(1)
    total_moves = len(opponent_history)

    # Thresholds for identifying strategies
    if defect_count / total_moves > 0.8:  # Opponent defects most of the time
        return 1  # Always defect
    elif cooperate_count / total_moves > 0.8:  # Opponent cooperates most of the time
        return 1  # Exploit by defecting
    elif opponent_history[-2:] == [1, 1]:  # Tit-for-2-tats detection
        return 1
    elif opponent_history[-1] == 1:  # Tit-for-tat detection
        return 1  # Defect in response
    else:
        return 0  # Cooperate otherwise
    
class ActorCriticRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=1):
        super(ActorCriticRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.actor = nn.Linear(hidden_size, 2)  # Outputs probabilities for cooperate/defect
        self.critic = nn.Linear(hidden_size, 1)  # Estimates state value

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            hidden = (h0, c0)
        out, hidden = self.lstm(x, hidden)
        last_out = out[:, -1, :]  # Use the last output of the sequence
        action_probs = torch.softmax(self.actor(last_out), dim=-1)
        state_value = self.critic(last_out)
        return action_probs, state_value, hidden

# Initialize the model (Note: Load pre-trained weights in a real scenario)
model = ActorCriticRNN()
model.eval()  # Set to evaluation mode

def agent_player(player_history, opponent_history):
    window_size = 5  # Number of past moves to consider
    # Combine histories into pairs of (agent_action, opponent_action)
    combined = list(zip(player_history, opponent_history))
    # Pad or truncate the history to the window size
    if len(combined) < window_size:
        combined = [(0, 0)] * (window_size - len(combined)) + combined
    else:
        combined = combined[-window_size:]
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
    # Get action probabilities from the model
    with torch.no_grad():
        action_probs, _, _ = model(input_tensor)
    # Select action with the highest probability
    return torch.argmax(action_probs).item()



# player1 = Player(strategy=agent_player)  
# player2 = Player(strategy=pavlov)  

# game = Game(trials=100)
# score_player1, score_player2 = game.play(player1, player2)

# # print(player1.history)
# # print(player2.history)

# print(f"Player 1's score: {score_player1}")
# print(f"Player 2's score: {score_player2}")

# winner =  1
# if score_player1 < score_player2:
#     winner = 2

# if score_player1==score_player2:
#     print(f"Game tied")
# else:
#     print(f"WINNER: player {winner}")