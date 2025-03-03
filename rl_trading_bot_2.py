# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tqdm import tqdm
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import os
import glob
import copy

# -------------------------------------
# Enhanced Technical Indicators
# -------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def compute_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

class EnhancedTradingEnv(gym.Env):
    def __init__(self, data, look_back=30, initial_balance=10000, drawdown_threshold=0.5):
        super(EnhancedTradingEnv, self).__init__()
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.drawdown_threshold = drawdown_threshold
        self.original_data = data.copy()
        
        # Feature pipeline
        self._prepare_features()
        self._normalize_features()
        
        self.prices = self.original_data['close'].values
        self.position_history = deque(maxlen=look_back)
        self.action_space = spaces.Discrete(3)  # 0: close, 1: long, 2: short
        self.observation_space = spaces.Box(
            low=-5, high=5,
            shape=(look_back, len(self.features) + 1),
            dtype=np.float32
        )
        self.stop_loss_pct = 0.05
        self.trade_duration = 0
        self.trade_log = []
        self.min_hold_period = 5
        self.trade_cooldown = 0
        self.min_cooldown = 2
        self.position_entry_step = 0
        self.peak_value = initial_balance
        self.positive_trades = 0
        self.total_trades = 0

    def _prepare_features(self):
        df = self.original_data
        
        # Existing features
        df['sma10'] = df['close'].rolling(window=10).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['rsi'] = compute_rsi(df['close'])
        
        # New features
        df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = compute_bollinger_bands(df['close'])
        df['atr'] = compute_atr(df['high'], df['low'], df['close'])
        df['volatility'] = df['close'].rolling(window=14).std()
        
        # Position-based features
        df['price_change'] = df['close'].pct_change()
        df['volume_trend'] = df['volume'].pct_change().rolling(window=5).mean()
        
        df = df.dropna()
        self.original_data = df
        
        self.features = [
            'close', 'sma10', 'sma50', 'sma200', 'ema10', 'rsi',
            'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle',
            'bb_lower', 'atr', 'volatility', 'price_change', 'volume_trend'
        ]
        self.raw_data = df[self.features]

    def _normalize_features(self):
        train_cutoff = int(0.8 * len(self.raw_data))
        train_data = self.raw_data.iloc[:train_cutoff]
        
        self.mean = train_data.mean()
        self.std = train_data.std().replace(0, 1e-6)
        self.normalized_data = (self.raw_data - self.mean) / self.std
        self.data_array = self.normalized_data.values.astype(np.float32)

    def reset(self):
        self.current_step = self.look_back
        self.cash = self.initial_balance
        self.position = 0
        self.position_size = 0
        self.entry_price = 0.0
        self.position_history.extend([0] * self.look_back)
        self.trade_duration = 0
        self.trade_count = 0
        self.trade_cooldown = 0
        self.position_entry_step = 0
        self.peak_value = self.initial_balance
        self.positive_trades = 0
        self.total_trades = 0
        return self._get_observation()

    def _get_observation(self):
        market_data = self.data_array[self.current_step-self.look_back:self.current_step]
        positions = np.array(self.position_history).reshape(-1,1).astype(np.float32)
        return np.concatenate([market_data, positions], axis=1)

    def step(self, action):
        prev_value = self.cash + self.position * self.entry_price * self.position_size
        price = self.prices[self.current_step]
        previous_position = self.position
        
        if self.trade_cooldown > 0:
            self.trade_cooldown -= 1
        
        # Enhanced stop loss check
        stop_loss_triggered = False
        if self.position != 0:
            stop_price = self.entry_price * (1 - self.stop_loss_pct) if self.position == 1 else \
                        self.entry_price * (1 + self.stop_loss_pct)
            stop_loss_triggered = (self.position == 1 and price <= stop_price) or \
                                (self.position == -1 and price >= stop_price)
        
        # Force close if stop loss hit
        if stop_loss_triggered:
            if self.position == 1:
                self._close_long(price)
                action_result = 'stop loss close long'
            else:
                self._close_short(price)
                action_result = 'stop loss close short'
            # Override any other action
            action = 0
            self.trade_cooldown = self.min_cooldown
        
        if self.position != 0 and self.current_step - self.position_entry_step < self.min_hold_period:
            action = 0
        
        action_result = self._process_action(action, price)
        new_value = self.cash + self.position * price * self.position_size
        
        self.trade_duration = self.trade_duration + 1 if self.position != 0 else 0
        position_duration = self.current_step - self.position_entry_step if self.position != 0 else 0
        
        reward_components = self._calculate_rewards(prev_value, new_value, previous_position, 
                                                  action, position_duration)
        
        current_value = self.cash + self.position * self.position_size * price
        self.peak_value = max(self.peak_value, current_value)
        net_drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
        
        done = net_drawdown >= self.drawdown_threshold or self.current_step >= len(self.data_array) - 1
        reward = reward_components['total']
        
        self._record_trade(action_result, price, self.position_size, new_value, reward_components)
        
        self.current_step += 1
        self.position_history.append(self.position)
        
        info = {
            'value': current_value,
            'position': self.position,
            'total_trades': self.total_trades,
            'positive_trades': self.positive_trades,
            'peak_value': self.peak_value,
            'reward_components': reward_components
        }
        
        return self._get_observation(), reward, done, info

    def _check_stop_loss(self, price):
        if self.position == 0:
            return False
        stop_price = self.entry_price * (1 - self.stop_loss_pct) if self.position == 1 else \
                    self.entry_price * (1 + self.stop_loss_pct)
        return price <= stop_price if self.position == 1 else price >= stop_price

    def _record_trade(self, action_result, price, quantity, networth, reward_components):
        trade_record = {
            'timestamp': self.original_data.iloc[self.current_step]['datetime'],
            'action': action_result,
            'price': float(price),
            'quantity': float(quantity),
            'networth': float(networth),
            'reward_components': reward_components.copy()
        }
        self.trade_log.append(trade_record)
        if 'close' in action_result and 'stop loss' not in action_result:
            self.total_trades += 1
            if networth > self.initial_balance:
                self.positive_trades += 1

    def _process_action(self, action, price):
        if self.trade_cooldown > 0:
            return 'cooldown'
        
        atr = self.original_data['atr'].iloc[self.current_step]
        position_size = self._calculate_position_size(price, atr)
            
        if action == 0 and self.position != 0:
            if self.position == 1:
                self._close_long(price)
                return 'close long'
            else:
                self._close_short(price)
                return 'close short'
        
        elif action == 1 and self.position == -1:
            self._close_short(price)
            self._open_long(price, position_size)
            return 'close short->buy'
        
        elif action == 2 and self.position == 1:
            self._close_long(price)
            self._open_short(price, position_size)
            return 'close long->sell'
                
        elif action == 1 and self.position == 0:
            self._open_long(price, position_size)
            return 'buy'
            
        elif action == 2 and self.position == 0:
            self._open_short(price, position_size)
            return 'sell'
            
        return 'hold long' if self.position == 1 else 'hold short' if self.position == -1 else 'no action'

    def _calculate_position_size(self, price, atr):
        risk_per_trade = self.cash * 0.01  # 1% risk
        stop_loss_pips = self.stop_loss_pct * price
        return min(self.initial_balance * 0.1, risk_per_trade / (atr * 2)) if atr > 0 else self.initial_balance * 0.01

    def _open_long(self, price, size):
        self.position_size = size
        self.cash -= self.position_size * price
        self.position = 1
        self.entry_price = price
        self.position_entry_step = self.current_step
        self.trade_cooldown = self.min_cooldown

    def _close_long(self, price):
        self.cash += self.position_size * price
        self.position = 0
        self.position_size = 0
        self.trade_cooldown = self.min_cooldown

    def _open_short(self, price, size):
        self.position_size = size
        self.cash += self.position_size * price
        self.position = -1
        self.entry_price = price
        self.position_entry_step = self.current_step
        self.trade_cooldown = self.min_cooldown

    def _close_short(self, price):
        self.cash -= self.position_size * price
        self.position = 0
        self.position_size = 0
        self.trade_cooldown = self.min_cooldown

    def _calculate_rewards(self, prev_value, new_value, previous_position, action, position_duration):
        rc = {}
        current_price = self.prices[self.current_step]
        
        # 1. Core PnL Calculation (40% weight)
        raw_return = (new_value - prev_value) / (self.initial_balance + 1e-6)
        rc['pnl'] = np.clip(raw_return * 10, -2.5, 2.5)  # Scaled to ±2.5
        
        # 2. Risk-Adjusted Return (30% weight)
        volatility = self.original_data['volatility'].iloc[self.current_step] + 1e-6
        rc['sharpe'] = np.clip(raw_return / volatility * 3, -1.5, 1.5)
        
        # 3. Position Management (20% weight)
        rc['time_penalty'] = -0.01 if self.position != 0 else 0  # Encourage closing
        rc['duration_bonus'] = min(0.5, position_duration * 0.02)  # +0.02 per step
        
        # 4. Drawdown Control (10% weight)
        current_drawdown = (self.peak_value - new_value) / (self.peak_value + 1e-6)
        rc['drawdown_penalty'] = -current_drawdown * 3  # Linear penalty
        
        # 5. Stop Loss Awareness
        if self.position != 0:
            unrealized_pct = (current_price - self.entry_price)/self.entry_price
            rc['risk_penalty'] = -abs(unrealized_pct) * 0.5  # Penalize floating losses
            
        # Normalize and sum components
        total_reward = sum(rc.values())
        return {'total': total_reward, **rc}

# -------------------------------------
# LSTM-Attention Network
# -------------------------------------
class LSTMAttentionDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMAttentionDQN, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # Use last timestep
        out = self.fc(attn_out[:, -1, :])
        return out

# -------------------------------------
# Training Manager
# -------------------------------------
class TrainingManager:
    def __init__(self, data_path, drawdown_threshold=0.5):
        self.data = pd.read_csv(data_path, parse_dates=['datetime']).sort_values('datetime')
        self.env = EnhancedTradingEnv(self.data, drawdown_threshold=drawdown_threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_dim = self.env.observation_space.shape[1]
        self.policy_net = LSTMAttentionDQN(input_dim, 128, self.env.action_space.n).to(self.device)
        self.target_net = LSTMAttentionDQN(input_dim, 128, self.env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.0005, weight_decay=1e-5)
        self.scaler = GradScaler()
        self.replay_buffer = deque(maxlen=20000)
        self.writer = SummaryWriter('logs')
        self.batch_size = 64
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.best_networth = -np.inf
        self.best_model_weights = None
        self.target_update_freq = 10
        self.last_best_networth = -np.inf
        self.checkpoint_counter = 0
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, episodes=200):
        torch.backends.cudnn.benchmark = True
        epsilon = 1.0
        epsilon_min = 0.2
        epsilon_decay = 0.98

        pbar = tqdm(total=episodes, desc="Training Progress")

        for episode in range(episodes):
            total_trades = 0
            positive_rewards = 0
            log_file = open(f'episode_{episode}.csv', 'w', encoding='utf-8')
            log_file.write("timestamp,action,quantity,price,networth,reward_total,pnl_bonus,time_bonus,duration_bonus,early_penalty,drawdown_penalty\n")
            
            state = self.env.reset()
            total_reward = 0
            done = False
            step = 0  # Initialize step counter
            max_steps = len(self.env.data_array) - self.env.look_back - 1

            while not done and step < max_steps:
                # Epsilon-greedy action selection
                valid_actions = [0]
                if self.env.position == 0:
                    valid_actions += [1, 2]
                elif self.env.position == 1:
                    valid_actions.append(2)
                else:
                    valid_actions.append(1)

                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    action = self._get_network_action(state, valid_actions)

                # Execute action
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                self.replay_buffer.append((state, action, reward, next_state, done))
                
                # CRITICAL: Increment step counter
                step += 1

                # Logging and optimization
                for trade in self.env.trade_log:
                    rc = trade['reward_components']
                    log_line = (f"{trade['timestamp']},"
                                f"{trade['action']},"
                                f"{trade['quantity']:.2f},"
                                f"{trade['price']:.2f},"
                                f"{trade['networth']:.2f},"
                                f"{rc['total']:.4f},"
                                f"{rc['pnl']:.4f},"
                                f"{rc['time_penalty']:.4f},"
                                f"{rc['duration_bonus']:.4f},"
                                f"{rc.get('risk_penalty',0):.4f},"
                                f"{rc['drawdown_penalty']:.4f}\n")
                    log_file.write(log_line)
                self.env.trade_log.clear()
                
                self._optimize_model()


                # Update progress bar every 30 steps
                if step % 30 == 0:
                    pbar.set_postfix({
                        'Step': f"{step}",
                        'Reward': f"{total_reward:.2f}",
                        'Epsilon': f"{epsilon:.2f}",
                        'NetWorth': f"{info['value']:.2f}",
                        'Pos': self.env.position,
                        'Trades': total_trades,
                        'Win%': f"{(positive_rewards/total_trades*100):.1f}%" if total_trades > 0 else "N/A"
                    })

                if done:
                    break

            # End of episode updates
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            log_file.close()
            pbar.update(1)

            # Update target network
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())


            if episode % 10 == 0:
                self._save_checkpoint(episode)

        pbar.close()

    def _optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_net(next_states).max(1)[0].detach()
            targets = rewards + (1 - dones) * self.gamma * next_q_values
            
            loss = nn.SmoothL1Loss()(q_values.squeeze(), targets)
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _get_network_action(self, state, valid_actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]
        
        for action in range(3):
            if action not in valid_actions:
                q_values[action] = -float('inf')
        
        return torch.argmax(q_values).item()

    def _save_checkpoint(self, episode):
        """Save model checkpoint with episode number"""
        checkpoint_path = f"{self.checkpoint_dir}/episode_{episode:04d}.pt"
        torch.save({
            'episode': episode,
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_networth': self.best_networth
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

# -------------------------------------
# Main Execution
# -------------------------------------
if __name__ == '__main__':
    # Clean up old files
    for fname in glob.glob("episode_*.txt"):
        os.remove(fname)
        print(f"Deleted old episode file: {fname}")
        
    if os.path.exists('logs'):
        for f in glob.glob('logs/events.out.tfevents*'):
            os.remove(f)
        print("Cleared TensorBoard logs")
        
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Torch version:", torch.__version__)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")

    data_path = './BTCUSDT.data'
    manager = TrainingManager(data_path, drawdown_threshold=0.3)
    manager.train(episodes=200)