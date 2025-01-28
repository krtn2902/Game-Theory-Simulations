import random
import torch
import torch.nn as nn
import torch.optim as optim

# Define the RNN-based Actor-Critic model
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

# Existing game and strategy code (unchanged)
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
        if move_player1 == 0 and move_player2 == 0:
            self.player1_score += 3
            self.player2_score += 3
        elif move_player1 == 0 and move_player2 == 1:
            self.player1_score += 0
            self.player2_score += 5
        elif move_player1 == 1 and move_player2 == 0:
            self.player1_score += 5
            self.player2_score += 0
        else:
            self.player1_score += 1
            self.player2_score += 1

# Strategy functions (unchanged)
def tit_for_tat(player_history, opponent_history):
    return opponent_history[-1] if opponent_history else 0

def grim(player_history, opponent_history):
    if 1 in opponent_history or (player_history and player_history[-1] == 1):
        return 1
    return 0

# Example usage
if __name__ == "__main__":
    # Create players
    rl_agent = Player(strategy=agent_player)
    opponent = Player(strategy=tit_for_tat)

    # Play a game
    game = Game(trials=10)
    score_rl, score_opponent = game.play(rl_agent, opponent)

    print(f"RL Agent's score: {score_rl}")
    print(f"Opponent's score: {score_opponent}")
    print("RL Agent's moves:", rl_agent.history)
    print("Opponent's moves:", opponent.history)