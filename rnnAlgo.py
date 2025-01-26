import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the payoff matrix
PAYOFF_MATRIX = {
    (0, 0): 3,  # Both cooperate
    (0, 1): 0,  # Cooperate vs defect
    (1, 0): 5,  # Defect vs cooperate
    (1, 1): 1   # Both defect
}

class PolicyNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, output_size=2):
        super(PolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.hidden_size = hidden_size
        self.hidden = None

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]  # Take the last output of the sequence
        fc_out = self.fc(lstm_out)
        probs = self.softmax(fc_out)
        return probs, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

def tit_for_tat(history):
    if not history:
        return 0  # Cooperate first
    else:
        return history[-1][0]  # Mirror opponent's last action

def always_defect(history):
    return 1

def random_strategy(history):
    return np.random.choice([0, 1])

def calculate_reward(my_action, opponent_action):
    return PAYOFF_MATRIX[(my_action, opponent_action)]

def discount_rewards(rewards, gamma=0.99):
    discounted = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    discounted = torch.tensor(discounted)
    return (discounted - discounted.mean()) / (discounted.std() + 1e-9)

def play_episode(model, opponent_strategy, num_rounds=10, training=True):
    model.hidden = model.init_hidden()
    history = []
    log_probs = []
    rewards = []
    
    # Initial previous actions (batch_size=1, seq_len=1, input_size=2)
    prev_actions = torch.zeros(1, 1, 2) if training else None

    for _ in range(num_rounds):
        probs, hidden = model(prev_actions, model.hidden)
        model.hidden = hidden  # Update hidden state
        
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        
        opponent_action = opponent_strategy(history)
        reward = calculate_reward(action.item(), opponent_action)
        
        log_probs.append(log_prob)
        rewards.append(reward)
        history.append((action.item(), opponent_action))
        
        # Prepare next input: (my_action, opponent_action)
        prev_actions = torch.tensor([[[action.item(), opponent_action]]], dtype=torch.float32)
    
    return log_probs, rewards

def train(model, optimizer, num_episodes=1000, gamma=0.99):
    for episode in range(num_episodes):
        # Vary opponents for robustness
        if episode % 4 == 0:
            opponent = tit_for_tat
        elif episode % 4 == 1:
            opponent = always_defect
        elif episode % 4 == 2:
            opponent = random_strategy
        else:
            opponent = np.random.choice([tit_for_tat, always_defect, random_strategy])
        
        log_probs, rewards = play_episode(model, opponent)
        discounted_rewards = discount_rewards(rewards, gamma)
        
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            avg_reward = sum(rewards) / len(rewards)
            print(f"Episode {episode}, Loss: {policy_loss.item():.2f}, Avg Reward: {avg_reward:.2f}")

# Initialize model and optimizer
model = PolicyNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
train(model, optimizer, num_episodes=1000)

# Example of using the trained model against Tit-for-Tat
def test_model(model, opponent_strategy, num_rounds=10):
    model.eval()
    history = []
    total_reward = 0
    
    model.hidden = model.init_hidden()
    prev_actions = torch.zeros(1, 1, 2)
    
    for _ in range(num_rounds):
        with torch.no_grad():
            probs, hidden = model(prev_actions, model.hidden)
            model.hidden = hidden
            action = torch.argmax(probs).item()
        
        opponent_action = opponent_strategy(history)
        reward = calculate_reward(action, opponent_action)
        total_reward += reward
        
        history.append((action, opponent_action))
        prev_actions = torch.tensor([[[action, opponent_action]]], dtype=torch.float32)
    
    print(f"Test against {opponent_strategy.__name__}:")
    print(f"Total reward: {total_reward}, Actions: {[a[0] for a in history]}")

# Test against different strategies
test_model(model, tit_for_tat)
test_model(model, always_defect)
test_model(model, random_strategy)