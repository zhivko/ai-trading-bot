# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces, Env
import torch
import torch.nn as nn
import os
import glob
from tqdm import tqdm
from torch.distributions import Categorical
import random

### PPO Agent Class
class PPOTrader:
    def __init__(self, env, hidden_size=256, policy_lr=0.0001, gamma=0.99, 
                 clip_epsilon=0.1, batch_size=512, ent_coef=0.02, gae_lambda=0.97):
        self.env = env
        self.base_env = env.env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Define RNN (LSTM) policy network
        input_size = env.observation_space['observation'].shape[0]
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_size, env.action_space.n).to(self.device)
        self.softmax = nn.Softmax(dim=-1).to(self.device)
        
        # Optimizer and hyperparameters
        self.optimizer = torch.optim.Adam(list(self.rnn.parameters()) + list(self.fc.parameters()), lr=policy_lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = ent_coef
        self.gae_lambda = gae_lambda

    def train(self, total_steps=1_000_000):
        pbar = tqdm(total=total_steps, desc="Training ProTrader")
        steps_taken = 0
        
        self.base_env.enable_episode_saving(True)
        
        while steps_taken < total_steps:
            obs, _ = self.env.reset()
            peak_networth = self.base_env.initial_balance
            epsilon = 0.5
            min_epsilon = 0.1
            epsilon_decay = 0.9995
            
            batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_dones = [], [], [], [], []
            hidden = (torch.zeros(1, 1, 256).to(self.device), torch.zeros(1, 1, 256).to(self.device))
            
            done = False
            while not done and steps_taken < total_steps:
                action_mask_np = obs['action_mask']
                state_tensor = torch.FloatTensor(obs['observation']).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, input_size]
                
                # Get action probabilities
                with torch.no_grad():
                    out, hidden = self.rnn(state_tensor, hidden)
                    logits = self.fc(out.squeeze(0))
                    probs = self.softmax(logits) * torch.FloatTensor(action_mask_np).to(self.device)
                    probs = probs / (probs.sum() + 1e-8)
                    dist = Categorical(probs)
                    if random.random() < epsilon:
                        valid_actions = np.where(action_mask_np)[0]
                        action = torch.tensor(np.random.choice(valid_actions), device=self.device)
                    else:
                        action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                # Step environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                if self.base_env.current_step % 10 == 0 or done:
                    self.base_env._save_episode()
                
                # Collect experience
                batch_obs.append(state_tensor.squeeze(0))
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_rewards.append(reward)
                batch_dones.append(done)
                
                current_networth = self.base_env.get_networth()
                peak_networth = max(peak_networth, current_networth)
                
                if steps_taken % 50 == 0:
                    pbar.set_postfix({
                        'networth': f'{current_networth:.2f}',
                        'peak': f'{peak_networth:.2f}',
                        'steps': steps_taken
                    })
                
                steps_taken += 1
                obs = next_obs
                
                # Update policy when batch is full or episode ends
                if len(batch_obs) >= self.batch_size or done:
                    states_batch = torch.cat(batch_obs)
                    actions_batch = torch.tensor(batch_actions, device=self.device)
                    old_log_probs_batch = torch.stack(batch_log_probs)
                    rewards_batch = torch.tensor(batch_rewards, device=self.device)
                    dones_batch = torch.tensor(batch_dones, device=self.device, dtype=torch.float32)
                    
                    returns, advantages = self._calculate_gae(rewards_batch, dones_batch, states_batch)
                    loss = self._compute_loss(states_batch, actions_batch, old_log_probs_batch, returns, advantages)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.rnn.parameters()) + list(self.fc.parameters()), 0.5)
                    self.optimizer.step()
                    
                    batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_dones = [], [], [], [], []
                
                epsilon = max(epsilon * epsilon_decay, min_epsilon)
            
            self.base_env._save_episode()
            pbar.update(self.base_env.current_step)
        
        self.base_env._save_episode()
        pbar.close()

    def _calculate_gae(self, rewards, dones, states):
        returns = torch.zeros_like(rewards, device=self.device)
        advantages = torch.zeros_like(rewards, device=self.device)
        next_value = 0
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t])
            advantages[t] = delta + self.gamma * self.gae_lambda * advantage * (1 - dones[t])
            returns[t] = advantages[t]
            next_value = 0
            advantage = advantages[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _compute_loss(self, states, actions, old_log_probs, returns, advantages):
        states = states.unsqueeze(1)
        hidden = (torch.zeros(1, states.size(0), 256).to(self.device), 
                  torch.zeros(1, states.size(0), 256).to(self.device))
        out, _ = self.rnn(states, hidden)
        logits = self.fc(out.squeeze(1))
        action_probs = self.softmax(logits)
        
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        total_loss = policy_loss - self.entropy_coeff * entropy
        
        return total_loss

### Technical Indicator Functions
def compute_donchian_channels(df, window=20):
    df['dc_upper'] = df['high'].rolling(window).max()
    df['dc_lower'] = df['low'].rolling(window).min()
    return df

def compute_atr(high, low, close, window=14):
    tr = pd.DataFrame({
        'tr1': high - low,
        'tr2': abs(high - close.shift()),
        'tr3': abs(low - close.shift())
    }).max(axis=1)
    return tr.rolling(window).mean()

def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    df['timestamp'] = df['datetime'].astype('int64') // 10**9
    df = compute_donchian_channels(df, window=20)
    df['atr'] = compute_atr(df['high'], df['low'], df['close'], window=14)
    df['rsi'] = compute_rsi(df['close'], window=14)
    df['ema_short'] = df['close'].ewm(span=12).mean()
    df['ema_long'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_short'] - df['ema_long']
    df['signal'] = df['macd'].ewm(span=9).mean()
    
    for col in ['close', 'atr', 'rsi', 'macd', 'signal']:
        df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()
    
    return df.dropna()

### Trading Environment Class
class ProTraderEnv(Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df
        self.max_steps = len(df)
        self.features = ['close_norm', 'atr_norm', 'rsi_norm', 'macd_norm', 'signal_norm']
        
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(5,), dtype=np.uint8)
        })
        
        self.action_space = spaces.Discrete(5)  # 0=hold, 1=buy, 2=close_long, 3=sell, 4=close_short
        
        self.initial_balance = initial_balance
        self.stop_loss = -0.03
        self.take_profit = 0.05
        self.position_size_fraction = 0.1
        
        self.episode_counter = 0
        self.episode_data = []
        self.save_episodes = False
        self.action_map = {
            0: "hold",
            1: "buy",
            2: "close_long",
            3: "sell",
            4: "close_short"
        }

    def enable_episode_saving(self, enable=True):
        self.save_episodes = enable
        return self

    def _get_action_mask(self):
        mask = np.zeros(5, dtype=np.uint8)
        if not self.position:
            mask[[0, 1, 3]] = 1
        elif self.position_type == 'long':
            mask[[0, 2]] = 1
        else:
            mask[[0, 4]] = 1
        return mask

    def _next_observation(self):
        clamped_step = min(self.current_step, len(self.df) - 1)
        obs = self.df[self.features].iloc[clamped_step].values.astype(np.float32)
        return {'observation': obs, 'action_mask': self._get_action_mask()}

    def reset(self, seed=None, options=None):
        if len(self.episode_data) > 0 and self.save_episodes:
            self._save_episode()
            self.episode_counter += 1
            self.episode_data = []
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = None
        self.position_size = 0
        self.position_type = None
        self.trade_duration = 0
        
        if seed is not None:
            np.random.seed(seed)
        
        return self._next_observation(), {}

    def get_networth(self):
        current_price = self.df['close'].iloc[self.current_step]
        if self.position_type == 'long':
            return self.balance + (self.position_size * current_price)
        elif self.position_type == 'short':
            return self.balance + ((self.entry_price - current_price) * self.position_size)
        return self.balance

    def step(self, action):
        if self.current_step >= self.max_steps - 1:
            return self._next_observation(), 0.0, True, False, {}

        # Calculate net worth before action
        old_networth = self.get_networth()
        current_price = self.df['close'].iloc[self.current_step]

        # Validate action; default to hold if invalid
        action_mask = self._get_action_mask()
        if action_mask[action] == 0:
            action = 0

        # Execute action
        if action == 1 and not self.position:
            self._open_position(current_price, 'long')
        elif action == 2 and self.position_type == 'long':
            self._close_position(current_price)
        elif action == 3 and not self.position:
            self._open_position(current_price, 'short')
        elif action == 4 and self.position_type == 'short':
            self._close_position(current_price)

        # Check stop loss or take profit
        if self.position:
            profit_pct = ((current_price - self.entry_price) / self.entry_price *
                          (1 if self.position_type == 'long' else -1))
            if profit_pct <= self.stop_loss or profit_pct >= self.take_profit:
                self._close_position(current_price)

        # Advance step
        self.current_step += 1
        self.trade_duration += 1 if self.position else 0

        # Calculate reward as change in net worth
        new_networth = self.get_networth()
        reward = new_networth - old_networth

        done = self.current_step >= self.max_steps - 1
        self._log_step(action, reward, done)

        return self._next_observation(), reward, done, False, {}

    def _log_step(self, action, reward, done):
        if not self.save_episodes or self.current_step >= len(self.df):
            return
            
        current_price = self.df['close'].iloc[self.current_step]
        networth = self.get_networth()
        
        log_entry = {
            'datetime': self.df['datetime'].iloc[self.current_step],
            'timestamp': self.df['timestamp'].iloc[self.current_step],
            'price': current_price,
            'action': self.action_map[action],
            'quantity': self.position_size,
            'balance': self.balance,
            'networth': networth,
            'reward': reward,
            'done': done
        }
        self.episode_data.append(log_entry)

    def _save_episode(self):
        if not self.save_episodes or not self.episode_data:
            return
            
        df = pd.DataFrame(self.episode_data)
        episode_filename = f"episode_{self.episode_counter:04d}.csv"
        df.to_csv(episode_filename, mode='a', header=not os.path.exists(episode_filename), index=False)
        self.episode_data = []

    def _open_position(self, price, position_type):
        self.position = price
        self.entry_price = price
        self.position_type = position_type
        self.position_size = (self.get_networth() * self.position_size_fraction) / price
        self.trade_duration = 0
        if position_type == 'long':
            self.balance -= price * self.position_size

    def _close_position(self, price):
        if self.position_type == 'long':
            self.balance += price * self.position_size
        elif self.position_type == 'short':
            profit = (self.entry_price - price) * self.position_size
            self.balance += profit
        self.position = None
        self.entry_price = None
        self.position_type = None
        self.position_size = 0
        self.trade_duration = 0

### Action Mask Wrapper
class CustomActionMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'observation': self.env.observation_space['observation'],
            'action_mask': spaces.Box(0, 1, (5,), dtype=np.uint8)
        })

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return {'observation': obs['observation'], 'action_mask': self.env._get_action_mask()}, info

### Main Execution
if __name__ == '__main__':
    # Clear old episode files
    for f in glob.glob("episode_*.csv"):
        os.remove(f)
        print(f"Removed old episode file: {f}")
    
    # Prepare data and train
    btc_data = preprocess_data('BTCUSDT.data')
    env = CustomActionMaskWrapper(ProTraderEnv(btc_data))
    env.env.enable_episode_saving(True)
    
    agent = PPOTrader(env)
    agent.train()