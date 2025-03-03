# -*- coding: utf-8 -*-

# conda env list
# conda activate tf-gpu
# tensorboard --logdir=logs --port=6007


# install nvidia cuda drivers: 
# make sure you select correct version!
#           https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
# install pytorch from https://pytorch.org/get-started/locally/
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



# -*- coding: utf-8 -*-

# Import required modules
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
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# -------------------------------------
# Helper Functions: Technical Indicators
# -------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_stochrsi(df, period=14, smoothK=3, smoothD=3):
    rsi = df['rsi']
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    return stoch_rsi.rolling(smoothK).mean(), stoch_rsi.rolling(smoothD).mean()

class EnhancedTradingEnv(gym.Env):
    def __init__(self, data, look_back=30, initial_balance=10000, drawdown_threshold=0.5):
        super(EnhancedTradingEnv, self).__init__()
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.drawdown_threshold = drawdown_threshold
        self.original_data = data.copy()
        self.data_array = None
        
        # Feature pipeline
        self._prepare_features()
        self._normalize_features()
        
        # Final validation
        assert self.data_array is not None, "data_array not initialized!"
        assert len(self.data_array) == len(self.prices), "Data/prices length mismatch!"
        
        self.prices = self.original_data['close'].values  # Uses cleaned data
        self.position_history = deque(maxlen=look_back)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-5, high=5,
            shape=(look_back, len(self.features) + 1),
            dtype=np.float32
        )
        self.position_size = 0
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.trade_duration = 0
        self.trade_log = []  # Add this line to store trade records
        
        # Add time constants
        self.min_hold_period = 5  # Minimum 6 steps (hours)
        self.day_steps = 24       # Steps per day
        self.week_steps = 168     # Steps per week
        self.trade_cooldown = 0  # Steps remaining before new trades allowed
        self.min_cooldown = 2    # Minimum steps between trades
        self.position_entry_step = 0  # Track when position was opened

    def _prepare_features(self):
        # Keep existing feature engineering
        df = self.original_data
        df['sma10'] = df['close'].rolling(10).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['rsi'] = compute_rsi(df['close'])
        df['stoch_k'], df['stoch_d'] = compute_stochrsi(df)
        df['ema_diff'] = df['EMA50'] - df['EMA200']
        df['ema_slope'] = df['EMA50'].diff().fillna(0)
        
        # Clean NaN values after feature calculations
        df = df.dropna().copy()  # Add this line to remove rows with any NaN values
        
        self.features = ['close', 'sma10', 'sma50', 'sma200', 
                        'EMA10', 'EMA50', 'EMA200', 'rsi', 
                        'stoch_k', 'stoch_d', 'ema_diff', 'ema_slope']
        self.raw_data = df[self.features]
        
        # After cleaning:
        self.original_data = df  # Update original_data with cleaned version
        self.prices = df['close'].values  # Store cleaned prices

    def _normalize_features(self):
        # Split into train/validation periods
        train_cutoff = int(0.8 * len(self.raw_data))
        train_data = self.raw_data.iloc[:train_cutoff]
        val_data = self.raw_data.iloc[train_cutoff:]
        
        # Calculate stats only on training data
        self.mean = train_data.mean()
        self.std = train_data.std()
        
        # Handle potential NaN/Inf values
        self.std = self.std.replace(0, 1e-6)  # Avoid division by zero
        self.normalized_data = (self.raw_data - self.mean) / self.std
        
        # Validation checks
        self._validate_normalization(train_data, val_data)
        
        # Convert to numpy array
        self.data_array = self.normalized_data.values.astype(np.float32)
        print(f"Data array shape: {self.data_array.shape}")  # Debug output

    def _validate_normalization(self, train_data, val_data):
        # Check 1: Training data should be ~N(0,1)
        normalized_train = (train_data - self.mean) / self.std
        print(f"\nNormalization Validation:")
        print(f"Train Mean: {normalized_train.mean().values.round(2)}")
        print(f"Train Std: {normalized_train.std().values.round(2)}")
        
        # Check 2: Validation data shouldn't be perfectly normalized
        normalized_val = (val_data - self.mean) / self.std
        print(f"\nValidation Mean: {normalized_val.mean().values.round(2)}")
        print(f"Validation Std: {normalized_val.std().values.round(2)}")
        
        # Check 3: No NaN values
        assert not self.normalized_data.isnull().values.any(), "NaNs in normalized data!"
        
        # Check 4: Reasonable value ranges
        outlier_count = (np.abs(self.normalized_data) > 5).sum().sum()
        print(f"Outlier count (|z|>5): {outlier_count}")

    def reset(self):
        self.current_step = self.look_back
        self.cash = self.initial_balance
        self.position = 0
        self.position_size = 0
        self.entry_price = 0.0
        self.position_history.extend([0] * self.look_back)
        self.trade_duration = 0
        self.trade_count = 0  # New trade counter
        self.trade_cooldown = 0
        self.position_entry_step = 0  # Track when position was opened
        return self._get_observation()
    
    def _get_observation(self):
        market_data = self.data_array[self.current_step-self.look_back:self.current_step]
        positions = np.array(self.position_history).reshape(-1,1).astype(np.float32)
        return np.concatenate([market_data, positions], axis=1)

    def step(self, action):
        #print(f"\nStep {self.current_step}")
        # Store previous portfolio value
        prev_value = self.cash + self.position * self.entry_price * self.position_size
        price = self.prices[self.current_step]
        previous_position = self.position
        
        # Store original action for logging
        original_action = action
        
        # Update cooldown counter EVERY STEP
        if self.trade_cooldown > 0:
            self.trade_cooldown -= 1
        
        # Enforce minimum holding period (modifies action)
        if self.position != 0: 
            position_duration = self.current_step - self.position_entry_step
            if position_duration < self.min_hold_period:
                action = 0  # Force hold if trying to close too early
            elif position_duration >= self.min_hold_period:
                # Allow closing but apply cooldown
                if action != 0:
                    self.trade_cooldown = self.min_cooldown
        
        # Process the MODIFIED action
        action_result = self._process_action(action, price)
        
        # Calculate new portfolio value
        new_value = self.cash + self.position * price * self.position_size
        
        # Update trade duration before calculating rewards
        if self.position != 0:
            self.trade_duration += 1
        else:
            self.trade_duration = 0
        
        # Calculate position duration BEFORE action
        position_duration = 0
        if self.position != 0:
            position_duration = self.current_step - self.position_entry_step
        
        # Pass duration to reward calculation
        reward_components = self._calculate_rewards(
            prev_value, new_value,
            previous_position, action,
            position_duration  # Now always initialized
        )
        
        # Calculate net worth and drawdown
        current_value = self.cash + self.position * self.position_size * price
        peak_value = max(self.initial_balance, current_value)
        net_drawdown = (peak_value - current_value) / peak_value
        
        # Early termination conditions
        done = False
        if net_drawdown >= self.drawdown_threshold:
            done = True
            reward_components['drawdown_penalty'] = -5.0
            print(f"\nDrawdown triggered: {net_drawdown:.2%} at step {self.current_step}")
        elif self.current_step >= len(self.data_array) - 1:
            done = True
        
        # Update reward with penalty BEFORE returning
        reward = sum(reward_components.values())
        
        # Record the trade with action result
        self._record_trade(
            action_result=action_result,
            price=price,
            quantity=self.position_size,
            networth=new_value,
            reward_components=reward_components
        )
        
        # Update environment state
        self.current_step += 1
        self.position_history.append(self.position)
        
        # Track position changes
        position_changed = (previous_position != self.position)
        
        # Update trade count
        if position_changed and (previous_position != 0 or self.position != 0):
            self.trade_count += 1
        
        # Package debug info
        info = {
            'value': new_value,
            'position': self.position,
            'action_result': action_result,
            'price': price,
            'reward_components': reward_components,
            'step': self.current_step,
            'original_action': original_action,
            'executed_action': action,
            'net_drawdown': net_drawdown
        }
        
        return self._get_observation(), reward, done, info

    def _record_trade(self, action_result, price, quantity, networth, reward_components):
        """Logs trade details with human-readable action labels"""
        action_map = {
            'cooldown': 'Cooldown',
            'close long': 'Close Long',
            'close short': 'Close Short',
            'buy': 'Buy',
            'sell': 'Sell',
            'close short->buy': 'Close Short -> Buy',
            'close long->sell': 'Close Long -> Sell',
            'hold long': 'Hold Long',
            'hold short': 'Hold Short',
            'no action': 'Hold',
            'error': 'Error'
        }
        
        trade_record = {
            'timestamp': self.original_data.iloc[self.current_step]['datetime'],
            'action': action_map.get(action_result, 'Unknown'),
            'price': float(price),
            'quantity': float(quantity),
            'networth': float(networth),
            'reward_components': reward_components.copy()
        }
        self.trade_log.append(trade_record)

    def _force_liquidate(self, price):
        if self.position == 1:
            self.cash += price * self.position_size
        elif self.position == -1:
            self.cash -= price * self.position_size
        self.position = 0
        self.position_size = 0

    def _process_action(self, action, price):
        """
    Process the agent's action while enforcing trading rules like cooldown and minimum hold period.
    
    Args:
        action (int): 0 (close position), 1 (buy/open long), 2 (sell/open short)
        price (float): Current price from the environment
        
    Returns:
        str: Result of the action (e.g., 'buy', 'close long', 'hold', 'cooldown')
    """
        # Check if in cooldown period
        if self.trade_cooldown > 0:
            return 'cooldown'
        
        try:
            result = 'no action'
            
            # Handle closing position (action 0)
            if action == 0 and self.position != 0:
                position_duration = self.current_step - self.position_entry_step
                if position_duration < self.min_hold_period:
                    # Prevent closing before minimum hold period; force hold instead
                    if self.position == 1:
                        result = 'hold long'
                    elif self.position == -1:
                        result = 'hold short'
                    #print(f"Prevented early close at step {self.current_step}: "
                    #    f"duration {position_duration} < {self.min_hold_period}")
                else:
                    # Close position if minimum hold period is met
                    if self.position == 1:
                        self._close_long(price)
                        result = 'close long'
                    elif self.position == -1:
                        self._close_short(price)
                        result = 'close short'
            
            # Handle composite actions (switching positions)
            elif action == 1 and self.position == -1:  # Close short before opening long
                self._close_short(price)
                self.trade_cooldown = self.min_cooldown  # Apply cooldown after closing
                result = 'close short'
            
            elif action == 2 and self.position == 1:  # Close long before opening short
                self._close_long(price)
                self.trade_cooldown = self.min_cooldown  # Apply cooldown after closing
                result = 'close long'
            
            # Handle new positions when no position exists
            elif action == 1 and self.position == 0:
                self._open_long(price)
                result = 'buy'
            
            elif action == 2 and self.position == 0:
                self._open_short(price)
                result = 'sell'
            
            # If action doesn't change position and we're holding, maintain hold
            elif action == 0 and self.position == 0:
                result = 'no action'
            elif action in [1, 2] and self.position != 0:
                # Ignore buy/sell attempts when already in a position (unless switching)
                if self.position == 1:
                    result = 'hold long'
                elif self.position == -1:
                    result = 'hold short'
            
            return result
        
        except Exception as e:
            print(f"Trade error: {e}")
            return 'error'


    def _open_long(self, price):
        max_position = self.cash / price
        self.position_size = min(max_position, self.initial_balance * 0.1)  # Risk management
        self.cash -= self.position_size * price
        self.position = 1
        self.entry_price = price
        self.position_entry_step = self.current_step  # Reset hold timer

    def _close_long(self, price):
        self.cash += self.position_size * price
        self.position = 0
        self.position_size = 0
        self.trade_cooldown = max(self.trade_cooldown, self.min_cooldown)

    def _open_short(self, price):
        max_position = self.cash / price
        self.position_size = min(max_position, self.initial_balance * 0.1)
        self.cash += self.position_size * price  # Credit for short
        self.position = -1
        self.entry_price = price
        self.position_entry_step = self.current_step  # Reset hold timer

    def _close_short(self, price):
        self.cash -= self.position_size * price  # Pay back short
        self.position = 0
        self.position_size = 0
        self.trade_cooldown = max(self.trade_cooldown, self.min_cooldown)

    def _calculate_rewards(self, prev_value, new_value, previous_position, executed_action, position_duration):
        """Calculate all reward components with exponential duration scaling"""
        rc = {
            'pnl': 0,
            'time_bonus': 0,
            'action_penalty': 0,
            'early_close_penalty': 0,
            'duration_bonus': 0,
            'profit_bonus': 0
        }
        
        # Initialize multipliers with safe defaults
        day_multiplier = 1.0
        week_multiplier = 1.0
        max_multiplier = 10.0
        scaled_day_multiplier = 1.0

        # 1. Enhanced PnL Calculation with increased scaling
        if prev_value != 0:
            if previous_position == 1:  # Long
                raw_pnl = (new_value - prev_value) / abs(prev_value)
            elif previous_position == -1:  # Short (FIXED DIRECTION)
                raw_pnl = (new_value - prev_value) / abs(prev_value)
            else:
                raw_pnl = 0

            # More aggressive asymmetric scaling
            if raw_pnl > 0:
                scaled_pnl = raw_pnl * 30 # Increased from 6.0
                scaled_pnl *= 1 + (position_duration/self.min_hold_period)**0.5
            else:
                scaled_pnl = raw_pnl * 30 # Increased from 3.0
            
            rc['pnl'] = np.clip(scaled_pnl, -4.0, 10.0)  # Wider bounds

        if previous_position != 0:
            hours_held = position_duration * 4
            days_held = hours_held // 24
            weeks_held = hours_held // 168
            
            rc['time_bonus'] = np.clip(0.01 * hours_held * (1 + days_held), 0, 5.0)  # Increase from 0.0005
            
            rc['duration_bonus'] = 0  # Reset to 0
            if hours_held >= 48:  # 2 days
                rc['duration_bonus'] = 1.0  # Significant bonus
            if hours_held >= 168:  # 1 week
                rc['duration_bonus'] += 2.0

        # 3. Early closure penalty (should trigger when closing before min hold)
        if previous_position != 0 and executed_action == 0:  # Only when closing
            position_duration = position_duration  # Duration before action
            if position_duration < self.min_hold_period:
                penalty = -2 * (self.min_hold_period - position_duration) ** 1.5
                rc['early_close_penalty'] = penalty
            
        # 4. Action penalty (small constant cost)
        rc['action_penalty'] = -0.002  # Base cost for any action

        # 5. Profit bonus now based on PNL
        if executed_action in [1, 2] and new_value > prev_value:
            rc['profit_bonus'] = np.clip(scaled_pnl * 0.5, -3.0, 3.0)  # Tied to PNL

        # Increase PNL weight in final reward
        total_reward = sum(rc.values()) + rc['pnl'] * 1.5  # Additional weighting
        
        return rc

# -------------------------------------
# Optimized DQN Network
# -------------------------------------
class ImprovedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),  # Increase hidden units
            nn.ReLU(),
            nn.Linear(256, 128),  # Add more layers
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------
# Optimized Training Manager
# -------------------------------------
class TrainingManager:
    def __init__(self, data_path, drawdown_threshold=0.5):
        self.epsilon = 1.0
        self.data = pd.read_csv(data_path, parse_dates=['datetime']).sort_values('datetime')
        self.env = EnhancedTradingEnv(self.data, drawdown_threshold=drawdown_threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        #self.policy_net = torch.compile(ImprovedDQN(input_dim, self.env.action_space.n).to(self.device))
        #self.target_net = torch.compile(ImprovedDQN(input_dim, self.env.action_space.n).to(self.device))
        self.policy_net = ImprovedDQN(input_dim, self.env.action_space.n).to(self.device)
        self.target_net = ImprovedDQN(input_dim, self.env.action_space.n).to(self.device)
        
        
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.001)  # Increased learning rate
        self.scaler = GradScaler()
        self.replay_buffer = deque(maxlen=10000)
        self.writer = SummaryWriter('logs')
        self.batch_size = 256
        self.gamma = 0.95  # Slightly shorter horizon
        self.positive_reward_scale = 2.0  # Increased positive reward emphasis
        self.priority_weight = 0.6  # Add this line - controls PER weighting (0.5-0.7 is typical)
        # Use pinned memory for data loading
        self.data_tensor = torch.tensor(self.env.data_array, dtype=torch.float32).pin_memory().to(self.device)

    def train(self, episodes=200):
        torch.backends.cudnn.benchmark = True
        epsilon = 1.0
        epsilon_min = 0.2
        epsilon_decay = 0.98

        pbar = tqdm(total=episodes, desc="Training Progress")

        for episode in range(episodes):
            total_trades = 0
            positive_trades = 0
            log_file = open(f'episode_{episode}.txt', 'w', encoding='utf-8')
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
                net_drawdown = info.get('net_drawdown', 0)
                total_reward += reward
                self.replay_buffer.append((state, action, reward, next_state, done))
                
                # CRITICAL: Increment step counter
                step += 1

                # ONLY LOG NEW TRADES AFTER ACTION EXECUTION
                if self.env.trade_log:  # Only write if trades occurred
                    for trade in self.env.trade_log:
                        total_trades += 1
                        if trade['reward_components'].get('pnl', 0) > 0:
                            positive_trades += 1
                        log_line = (f"{trade['timestamp']},"
                                    f"{trade['action']},"
                                    f"{trade['quantity']:.2f},"
                                    f"{trade['price']:.2f},"
                                    f"{trade['networth']:.2f},"
                                    f"{sum(trade['reward_components'].values()):.4f},"
                                    f"{trade['reward_components'].get('pnl',0):.4f},"
                                    f"{trade['reward_components'].get('time_bonus',0):.4f},"
                                    f"{trade['reward_components'].get('duration_bonus',0):.4f},"
                                    f"{trade['reward_components'].get('early_close_penalty',0):.4f},"
                                    f"{trade['reward_components'].get('drawdown_penalty',0):.4f}\n")
                        log_file.write(log_line)
                    self.env.trade_log.clear()  # Clear after writing
                
                self._optimize_model()

                # Update target network
                if episode % 10 == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # Update progress bar every 30 steps
                if step % 30 == 0:
                    pbar.set_postfix({
                        'Step': f"{step}",
                        'Reward': f"{total_reward:.2f}",
                        'Epsilon': f"{epsilon:.2f}",
                        'NetWorth': f"{info['value']:.2f}",
                        'Pos': self.env.position,
                        'Trades': total_trades,
                        'Win%': f"{(positive_trades/total_trades*100):.1f}%" if total_trades > 0 else "N/A"
                    })

                if done:
                    if net_drawdown >= self.env.drawdown_threshold:
                        print(f"Episode {episode} terminated with {net_drawdown:.2%} drawdown")                    
                    break

            pbar.update(1)

            # End of episode updates
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            log_file.close()


            if episode % 10 == 0:
                self._save_checkpoint(episode)

        pbar.close()

    def _save_checkpoint(self, episode):
        checkpoint = {
            'episode': episode,
            'model_state': self.policy_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'replay_buffer': list(self.replay_buffer)
        }
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/checkpoint_ep{episode}.pth')
        print(f"\nCheckpoint saved at episode {episode}")

    def _optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert NumPy arrays to PyTorch tensors
        states_tensor = torch.stack([torch.tensor(state, device=self.device) for state in states])
        next_states_tensor = torch.stack([torch.tensor(next_state, device=self.device) for next_state in next_states])
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            current_q = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))

            with torch.no_grad():
                next_q = self.target_net(next_states_tensor)
                next_q_max = next_q.max(1)[0]
                targets = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_max

            # Priority experience replay weighting
            positive_mask = rewards_tensor > 0
            weights = torch.where(positive_mask, 
                                torch.ones_like(rewards_tensor) * self.priority_weight,
                                torch.ones_like(rewards_tensor) * (1 - self.priority_weight))
            
            # Modified loss calculation
            loss = (weights * nn.SmoothL1Loss(reduction='none')(current_q.squeeze(), targets)).mean()
            
            # Scale gradients for positive experiences
            if positive_mask.any():
                self.scaler.scale(loss * self.positive_reward_scale).backward()
            else:
                self.scaler.scale(loss).backward()

        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def _get_network_action(self, state, valid_actions):
        state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]
        
        # Mask invalid actions
        for action in range(3):
            if action not in valid_actions:
                q_values[action] = -float('inf')
                
        return torch.argmax(q_values).item()

    def _update_model(self):
        # Sample from replay buffer
        states, actions, rewards, next_states = self._sample_batch()
        
        # Apply reward normalization
        rewards = self._normalize_rewards(rewards)
        
        # Continue with DQN update
        target_q = rewards + self.gamma * torch.max(self.target_net(next_states), dim=1)[0]

    def _normalize_rewards(self, rewards):
        """Apply reward scaling for training stability"""
        return (rewards - self.reward_mean) / (self.reward_std + 1e-8)

# -------------------------------------
# Main Execution
# -------------------------------------
if __name__ == '__main__':
    # Delete old episode log files
    for fname in glob.glob("episode_*.txt"):
        os.remove(fname)
        print(f"Deleted old episode file: {fname}")
    
    # Delete TensorBoard logs
    if os.path.exists('logs'):
        # Remove all event files
        for f in glob.glob('logs/events.out.tfevents*'):
            os.remove(f)
        # Optional: Remove entire directory
        print("Cleared TensorBoard logs")
    
    # Recreate directory if needed
    os.makedirs('logs', exist_ok=True)
    
    print("Torch version:", torch.__version__)
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")

    data_path = './BTCUSDT.data'
    manager = TrainingManager(data_path, drawdown_threshold=0.3)
    manager.train(episodes=200)
    torch.save(manager.policy_net.state_dict(), 'trained_model.pth')
    print("Training complete.")