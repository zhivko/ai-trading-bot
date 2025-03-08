# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces, Env
import os
import sys
import glob
from tqdm import tqdm
import random

# Set CUDA environment variables before importing torch
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if os.path.exists(cuda_path):
    os.environ["CUDA_HOME"] = cuda_path
    os.environ["CUDA_PATH"] = cuda_path
    
    # Add CUDA bin directory to PATH
    cuda_bin = os.path.join(cuda_path, "bin")
    if os.path.exists(cuda_bin):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]
    
    # Add CUDA lib directory to PATH
    cuda_lib = os.path.join(cuda_path, "lib", "x64")  # Use x64 for 64-bit Windows
    if os.path.exists(cuda_lib):
        os.environ["PATH"] = cuda_lib + os.pathsep + os.environ["PATH"]
    
    # Add cuDNN path if it exists
    cudnn_path = os.path.join(cuda_path, "extras", "CUPTI", "lib64")
    if os.path.exists(cudnn_path):
        os.environ["PATH"] = cudnn_path + os.pathsep + os.environ["PATH"]
    
    print(f"CUDA environment variables set to {cuda_path}")
    print(f"PATH now includes: {os.environ['PATH']}")
else:
    print(f"Warning: CUDA path {cuda_path} not found")

# Import torch after setting environment variables
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Minimal PyTorch version and CUDA information
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

### PPO Agent Class
class PPOTrader:
    def __init__(self, env, hidden_size=512, policy_lr=1e-4, gamma=0.99, 
                 clip_epsilon=0.2, batch_size=1024, ent_coef=0.01, gae_lambda=0.95,
                 sequence_length=10):  # Added sequence length parameter
        self.env = env
        self.base_env = env.env
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length  # Store sequence length for recurrent processing

        # Set device with minimal output
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Reduce batch size for better learning
        self.batch_size = batch_size
        
        # Get input size from environment
        input_size = env.observation_space['observation'].shape[0]
        
        # Create recurrent policy network with GRU
        self.policy_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        ).to(self.device)
        
        # GRU layer for policy
        self.policy_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        ).to(self.device)
        
        # Output layers for policy
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, env.action_space.n)
        ).to(self.device)
        
        # Create recurrent value network with GRU
        self.value_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        ).to(self.device)
        
        # GRU layer for value
        self.value_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        ).to(self.device)
        
        # Output layers for value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        ).to(self.device)
        
        # Use separate optimizers for policy and value networks
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.policy_gru.parameters()) + 
            list(self.policy_head.parameters()), 
            lr=policy_lr
        )
        
        self.value_optimizer = torch.optim.Adam(
            list(self.value_net.parameters()) + 
            list(self.value_gru.parameters()) + 
            list(self.value_head.parameters()), 
            lr=policy_lr * 2
        )
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = ent_coef
        self.gae_lambda = gae_lambda
        
        # Learning rate scheduler for adaptive learning
        self.policy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.policy_optimizer, mode='max', factor=0.5, patience=5, verbose=0
        )
        
        # Experience buffer for more stable learning
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_log_probs = []
        self.buffer_rewards = []
        self.buffer_dones = []
        
        # Observation history for recurrent processing
        self.obs_history = []
        
        # Tracking metrics
        self.episode_rewards = []
        self.best_reward = -float('inf')
        
        # Hidden states for recurrent networks
        self.policy_hidden = None
        self.value_hidden = None

    def train(self, total_steps=1_000_000):
        pbar = tqdm(total=total_steps, desc="Training ProTrader")
        steps_taken = 0
        episode_num = 0  # Track episode number
        
        self.base_env.enable_episode_saving(True)
        
        while steps_taken < total_steps:
            episode_num += 1  # Increment episode counter
            obs, _ = self.env.reset()
            peak_networth = self.base_env.initial_balance
            
            # Adaptive exploration
            epsilon = max(0.5 * (1 - episode_num / 100), 0.05)
            
            pbar.set_description(f"Episode {episode_num}")
            
            episode_reward = 0
            done = False
            
            # Reset observation history and hidden states at the start of each episode
            self.obs_history = []
            self.policy_hidden = None
            self.value_hidden = None
            
            while not done and steps_taken < total_steps:
                action_mask_np = obs['action_mask']
                
                # Add current observation to history
                self.obs_history.append(obs['observation'])
                
                # Keep only the last sequence_length observations
                if len(self.obs_history) > self.sequence_length:
                    self.obs_history = self.obs_history[-self.sequence_length:]
                
                # Create sequence tensor for recurrent processing
                if len(self.obs_history) < self.sequence_length:
                    # Pad with zeros if we don't have enough history
                    padding = [np.zeros_like(self.obs_history[0]) for _ in range(self.sequence_length - len(self.obs_history))]
                    sequence = padding + self.obs_history
                else:
                    sequence = self.obs_history
                
                # Convert sequence to tensor
                sequence_tensor = torch.FloatTensor(np.array(sequence)).unsqueeze(0).to(self.device)  # [1, seq_len, input_size]
                
                # Get action probabilities using recurrent policy network
                with torch.no_grad():
                    # Forward pass through feature extractor
                    features = self.policy_net(sequence_tensor.reshape(-1, sequence_tensor.size(-1)))
                    features = features.view(1, self.sequence_length, self.hidden_size)
                    
                    # Forward pass through GRU
                    if self.policy_hidden is None:
                        gru_out, self.policy_hidden = self.policy_gru(features)
                    else:
                        gru_out, self.policy_hidden = self.policy_gru(features, self.policy_hidden)
                    
                    # Get the last output (most recent time step)
                    gru_out = gru_out[:, -1]
                    
                    # Forward pass through policy head
                    logits = self.policy_head(gru_out).squeeze(0)
                    
                    # Apply action mask
                    masked_logits = logits.clone()
                    masked_logits[torch.tensor(action_mask_np, device=self.device) == 0] = float('-inf')
                    probs = torch.softmax(masked_logits, dim=0)
                    
                    # Epsilon-greedy exploration
                    if random.random() < epsilon:
                        valid_actions = np.where(action_mask_np)[0]
                        action = torch.tensor(np.random.choice(valid_actions), device=self.device)
                    else:
                        dist = Categorical(probs)
                        action = dist.sample()
                    
                    log_prob = torch.log(probs[action] + 1e-10)  # Add small epsilon to avoid log(0)
                
                # Step environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                
                # Save episode data periodically
                if self.base_env.current_step % 10 == 0 or done:
                    self.base_env._save_episode()
                
                # Store experience
                self.buffer_obs.append(obs['observation'])
                self.buffer_actions.append(action.item())
                self.buffer_log_probs.append(log_prob)
                self.buffer_rewards.append(reward)
                self.buffer_dones.append(done)
                
                # Track episode metrics
                episode_reward += reward
                current_networth = self.base_env.get_networth()
                peak_networth = max(peak_networth, current_networth)
                
                # Update progress bar
                if steps_taken % 50 == 0:
                    win_rate = self.base_env.profitable_trades / max(1, self.base_env.trade_count) * 100
                    pbar.set_postfix({
                        'episode': episode_num,
                        'networth': f'{current_networth:.2f}',
                        'peak': f'{peak_networth:.2f}',
                        'win_rate': f'{win_rate:.1f}%',
                        'epsilon': f'{epsilon:.2f}'
                    })
                
                steps_taken += 1
                obs = next_obs
                
                # Update policy when buffer is full
                if len(self.buffer_obs) >= self.batch_size:
                    self._update_policy()
                    # Reset hidden states after policy update
                    self.policy_hidden = None
                    self.value_hidden = None
            
            # End of episode updates
            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards[-100:])
            
            # Update learning rate based on performance
            self.policy_scheduler.step(avg_reward)
            
            # Save best model if performance improved
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self._save_model(f"best_model_ep{episode_num}.pt")
            
            # Save episode data
            self.base_env._save_episode()
            pbar.update(self.base_env.current_step)
            pbar.refresh()
        
        self.base_env._save_episode()
        pbar.close()
        
        # Save final model
        self._save_model("final_model.pt")

    def _update_policy(self):
        # Process experiences in sequences for recurrent networks
        sequences = []
        sequence_actions = []
        sequence_log_probs = []
        sequence_rewards = []
        sequence_dones = []
        
        # Create sequences from buffer
        for i in range(0, len(self.buffer_obs) - self.sequence_length + 1, self.sequence_length // 2):  # 50% overlap between sequences
            end_idx = min(i + self.sequence_length, len(self.buffer_obs))
            seq_len = end_idx - i
            
            if seq_len < 3:  # Skip sequences that are too short
                continue
                
            # Create sequence
            seq = self.buffer_obs[i:end_idx]
            act = self.buffer_actions[i:end_idx]
            log_p = self.buffer_log_probs[i:end_idx]
            rew = self.buffer_rewards[i:end_idx]
            don = self.buffer_dones[i:end_idx]
            
            # Pad sequences if needed
            if seq_len < self.sequence_length:
                pad_len = self.sequence_length - seq_len
                seq = [np.zeros_like(seq[0]) for _ in range(pad_len)] + seq
                act = [0 for _ in range(pad_len)] + act
                log_p = [torch.zeros(1, device=self.device) for _ in range(pad_len)] + log_p
                rew = [0 for _ in range(pad_len)] + rew
                don = [True for _ in range(pad_len)] + don
            
            sequences.append(seq)
            sequence_actions.append(act)
            sequence_log_probs.append(log_p)
            sequence_rewards.append(rew)
            sequence_dones.append(don)
        
        # Skip update if we don't have enough sequences
        if len(sequences) == 0:
            self.buffer_obs = []
            self.buffer_actions = []
            self.buffer_log_probs = []
            self.buffer_rewards = []
            self.buffer_dones = []
            return
        
        # Convert sequences to tensors
        states = torch.FloatTensor(np.array(sequences)).to(self.device)  # [num_seq, seq_len, input_size]
        actions = torch.tensor(sequence_actions, device=self.device)  # [num_seq, seq_len]
        
        # Stack log probs properly
        old_log_probs = []
        for seq_log_probs in sequence_log_probs:
            old_log_probs.append(torch.stack(seq_log_probs))
        old_log_probs = torch.stack(old_log_probs)  # [num_seq, seq_len]
        
        rewards = torch.tensor(sequence_rewards, device=self.device)  # [num_seq, seq_len]
        dones = torch.tensor(sequence_dones, device=self.device, dtype=torch.float32)  # [num_seq, seq_len]
        
        # Compute returns and advantages
        returns, advantages = self._compute_returns_and_advantages(rewards, dones, states)
        
        # Perform multiple epochs of training
        for _ in range(4):  # Increased from 3 to 4 epochs
            # Process each sequence through the recurrent networks
            batch_size = states.size(0)
            
            # Reshape for feature extraction
            states_flat = states.reshape(-1, states.size(-1))  # [batch_size * seq_len, input_size]
            
            # Forward pass through feature extractors
            policy_features = self.policy_net(states_flat)
            value_features = self.value_net(states_flat)
            
            # Reshape back to sequences
            policy_features = policy_features.view(batch_size, self.sequence_length, self.hidden_size)
            value_features = value_features.view(batch_size, self.sequence_length, self.hidden_size)
            
            # Forward pass through GRUs
            policy_out, _ = self.policy_gru(policy_features)  # [batch_size, seq_len, hidden_size]
            value_out, _ = self.value_gru(value_features)  # [batch_size, seq_len, hidden_size]
            
            # Forward pass through heads
            logits = self.policy_head(policy_out)  # [batch_size, seq_len, action_space]
            values = self.value_head(value_out).squeeze(-1)  # [batch_size, seq_len]
            
            # Reshape for loss calculation
            logits_flat = logits.reshape(-1, logits.size(-1))  # [batch_size * seq_len, action_space]
            actions_flat = actions.reshape(-1)  # [batch_size * seq_len]
            old_log_probs_flat = old_log_probs.reshape(-1)  # [batch_size * seq_len]
            advantages_flat = advantages.reshape(-1)  # [batch_size * seq_len]
            returns_flat = returns.reshape(-1)  # [batch_size * seq_len]
            values_flat = values.reshape(-1)  # [batch_size * seq_len]
            
            # Create mask for valid timesteps (not padding)
            valid_mask = ~torch.isnan(old_log_probs_flat)
            
            if valid_mask.sum() == 0:
                continue
                
            # Apply mask to get only valid timesteps
            logits_valid = logits_flat[valid_mask]
            actions_valid = actions_flat[valid_mask]
            old_log_probs_valid = old_log_probs_flat[valid_mask]
            advantages_valid = advantages_flat[valid_mask]
            returns_valid = returns_flat[valid_mask]
            values_valid = values_flat[valid_mask]
            
            # Calculate policy loss
            probs = torch.softmax(logits_valid, dim=1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions_valid)
            entropy = dist.entropy().mean()
            
            # PPO objective
            ratio = torch.exp(new_log_probs - old_log_probs_valid)
            surr1 = ratio * advantages_valid
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_valid
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = ((returns_valid - values_valid) ** 2).mean()
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_total_loss = policy_loss - self.entropy_coeff * entropy
            policy_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + 
                list(self.policy_gru.parameters()) + 
                list(self.policy_head.parameters()), 
                0.5
            )
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.value_net.parameters()) + 
                list(self.value_gru.parameters()) + 
                list(self.value_head.parameters()), 
                0.5
            )
            self.value_optimizer.step()
        
        # Clear buffer after update
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_log_probs = []
        self.buffer_rewards = []
        self.buffer_dones = []

    def _compute_returns_and_advantages(self, rewards, dones, states):
        # Compute value estimates for sequences
        batch_size = states.size(0)
        seq_len = states.size(1)
        
        with torch.no_grad():
            # Reshape for feature extraction
            states_flat = states.reshape(-1, states.size(-1))  # [batch_size * seq_len, input_size]
            
            # Forward pass through feature extractor
            value_features = self.value_net(states_flat)
            
            # Reshape back to sequences
            value_features = value_features.view(batch_size, seq_len, self.hidden_size)
            
            # Forward pass through GRU
            value_out, _ = self.value_gru(value_features)
            
            # Forward pass through value head
            values = self.value_head(value_out).squeeze(-1)  # [batch_size, seq_len]
        
        # Initialize returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Compute returns and advantages for each sequence
        for b in range(batch_size):
            gae = 0
            next_value = 0
            
            for t in reversed(range(seq_len)):
                if t == seq_len - 1:
                    next_value = 0
                else:
                    next_value = values[b, t + 1]
                
                # Skip padded timesteps
                if dones[b, t] and t < seq_len - 1:
                    continue
                
                delta = rewards[b, t] + self.gamma * next_value * (1 - dones[b, t]) - values[b, t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[b, t]) * gae
                
                returns[b, t] = gae + values[b, t]
                advantages[b, t] = gae
        
        # Normalize advantages within each sequence
        for b in range(batch_size):
            valid_indices = ~torch.isnan(advantages[b])
            if valid_indices.sum() > 1:
                adv_mean = advantages[b, valid_indices].mean()
                adv_std = advantages[b, valid_indices].std() + 1e-8
                advantages[b, valid_indices] = (advantages[b, valid_indices] - adv_mean) / adv_std
        
        return returns, advantages
    
    def _save_model(self, filename):
        """Save model weights to file"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'policy_gru_state_dict': self.policy_gru.state_dict(),
            'policy_head_state_dict': self.policy_head.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'value_gru_state_dict': self.value_gru.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load model weights from file"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            
            # Load network parameters
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.policy_gru.load_state_dict(checkpoint['policy_gru_state_dict'])
            self.policy_head.load_state_dict(checkpoint['policy_head_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.value_gru.load_state_dict(checkpoint['value_gru_state_dict'])
            self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
            
            # Load optimizer states
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            
            # Load hyperparameters
            self.sequence_length = checkpoint.get('sequence_length', 10)
            self.hidden_size = checkpoint.get('hidden_size', 512)
            
            print(f"Model loaded from {filename}")
            return True
        return False

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
    
    # Calculate all technical indicators
    df = compute_donchian_channels(df, window=20)
    df['atr'] = compute_atr(df['high'], df['low'], df['close'], window=14)
    df['rsi'] = compute_rsi(df['close'], window=14)
    df['ema_short'] = df['close'].ewm(span=12).mean()
    df['ema_long'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_short'] - df['ema_long']
    df['signal'] = df['macd'].ewm(span=9).mean()
    
    # Calculate additional features
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['atr'] / df['close']
    df['dc_width'] = (df['dc_upper'] - df['dc_lower']) / df['close']  # Normalized Donchian channel width
    
    # Enhance RSI and stochastic representation
    # Add RSI crossover signals and zones
    if 'rsi' in df.columns:
        # RSI zones (more explicit representation)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(float)  # Oversold zone
        df['rsi_overbought'] = (df['rsi'] > 70).astype(float)  # Overbought zone
        df['rsi_middle'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(float)  # Middle zone
        
        # RSI direction and momentum
        df['rsi_direction'] = df['rsi'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        df['rsi_acceleration'] = df['rsi'].diff().diff()
    
    # Add stochastic features if available
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        # Stochastic crossovers
        df['stoch_crossover'] = ((df['stoch_k'].shift(1) < df['stoch_d'].shift(1)) & 
                                (df['stoch_k'] > df['stoch_d'])).astype(float)
        df['stoch_crossunder'] = ((df['stoch_k'].shift(1) > df['stoch_d'].shift(1)) & 
                                 (df['stoch_k'] < df['stoch_d'])).astype(float)
        
        # Stochastic zones
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(float)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(float)
    
    # Normalize the new features
    features_to_normalize = [
        'open', 'high', 'low', 'close',  # Price data
        'atr', 'rsi', 'macd', 'signal',  # Technical indicators
        'ema_short', 'ema_long',         # Moving averages
        'dc_upper', 'dc_lower',          # Donchian channels
        'price_change', 'volatility', 'dc_width',  # Derived features
        # New features
        'rsi_direction', 'rsi_acceleration',
        'stoch_crossover', 'stoch_crossunder',
        'rsi_oversold', 'rsi_overbought', 'rsi_middle',
        'stoch_oversold', 'stoch_overbought'
    ]
    
    # Use rolling window normalization for more realistic training
    window_size = 100  # 100-period rolling window
    
    for col in features_to_normalize:
        if col in df.columns:
            rolling_mean = df[col].rolling(window=window_size).mean()
            rolling_std = df[col].rolling(window=window_size).std()
            df[f'{col}_norm'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)  # Add small epsilon to avoid division by zero
            
            # Forward fill NaN values at the beginning of the series
            df[f'{col}_norm'] = df[f'{col}_norm'].fillna(method='ffill')
            
            # If there are still NaN values (at the very beginning), fill with zeros
            df[f'{col}_norm'] = df[f'{col}_norm'].fillna(0)
    
    # Drop rows with NaN values
    return df.dropna()

### Trading Environment Class
class ProTraderEnv(Env):
    def __init__(self, df, initial_balance=10000, debug=False):
        super().__init__()
        self.df = df
        self.max_steps = len(df)
        
        # Expanded feature set with all normalized indicators
        self.features = [
            'close_norm', 'high_norm', 'low_norm', 'open_norm',  # Normalized price data
            'atr_norm', 'rsi_norm', 'macd_norm', 'signal_norm',  # Normalized technical indicators
            'ema_short_norm', 'ema_long_norm',                   # Normalized moving averages
            'dc_upper_norm', 'dc_lower_norm',                    # Normalized Donchian channels
            'price_change_norm', 'volatility_norm', 'dc_width_norm',  # Normalized derived features
            # Add new RSI and stochastic features
            'rsi_direction_norm', 'rsi_acceleration_norm',
            'rsi_oversold_norm', 'rsi_overbought_norm', 'rsi_middle_norm'
        ]
        
        # Add stochastic features if available
        stoch_features = [
            'stoch_crossover_norm', 'stoch_crossunder_norm',
            'stoch_oversold_norm', 'stoch_overbought_norm'
        ]
        
        # Only include stochastic features if they exist in the dataframe
        for feature in stoch_features:
            if feature in self.df.columns:
                self.features.append(feature)
        
        # Filter out features that don't exist in the dataframe
        self.features = [f for f in self.features if f in self.df.columns]
        
        print(f"Using {len(self.features)} features: {self.features}")
        
        self.debug = debug  # Debug flag
        
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(5,), dtype=np.uint8)
        })
        
        self.action_space = spaces.Discrete(5)  # 0=hold, 1=buy, 2=close_long, 3=sell, 4=close_short
        
        self.initial_balance = initial_balance
        self.take_profit = 0.09  # 9% take profit target
        self.stop_loss = -0.03   # 3% stop loss (1:3 risk-reward ratio)
        self.position_size_fraction = 0.2  # Increased from 0.1 for larger positions
        
        # Add trading fee to simulate real-world conditions
        self.trading_fee = 0.001  # 0.1% fee per trade
        
        # Add holding reward/penalty to discourage very short trades
        self.holding_reward = 0.0001  # Small reward for holding positions
        self.min_holding_period = 5  # Minimum number of steps to hold a position
        
        self.episode_counter = 0
        self.episode_data = []
        self.save_episodes = False
        self.current_episode_file = None  # Track current episode file
        self.action_map = {
            0: "hold",
            1: "buy",
            2: "close_long",
            3: "sell",
            4: "close_short"
        }
        
        # Track trade statistics
        self.trade_count = 0
        self.profitable_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        
        # Add reward scaling factors for trade outcomes
        self.positive_trade_multiplier = 2.0  # Amplify rewards for profitable trades
        self.negative_trade_multiplier = 2.5  # Amplify punishment for losing trades
        self.consecutive_loss_penalty = 0.2   # Additional penalty for consecutive losses
        
        # Track consecutive losses
        self.consecutive_losses = 0
        
        # Track trade performance
        self.trade_start_price = None
        self.trade_start_networth = None
        self.trade_unrealized_profit = 0.0

    def enable_episode_saving(self, enable=True):
        self.save_episodes = enable
        return self

    def _get_action_mask(self):
        mask = np.zeros(5, dtype=np.uint8)
        if not self.position:
            # If no position, can only hold, buy, or sell
            mask[[0, 1, 3]] = 1
        elif self.position_type == 'long':
            # If long position, can only hold or close long
            mask[[0, 2]] = 1
        elif self.position_type == 'short':
            # If short position, can only hold or close short
            mask[[0, 4]] = 1
        
        # Debug logging for action mask
        if self.debug and hasattr(self, 'episode_counter') and self.episode_counter > 0:
            position_status = f"Position: {self.position_type if self.position else 'None'}"
            mask_str = f"Mask: {mask} - Valid actions: {[i for i, m in enumerate(mask) if m == 1]}"
            print(f"Step {self.current_step}: {position_status} - {mask_str}")
            
        return mask

    def _next_observation(self):
        clamped_step = min(self.current_step, len(self.df) - 1)
        obs = self.df[self.features].iloc[clamped_step].values.astype(np.float32)
        return {'observation': obs, 'action_mask': self._get_action_mask()}

    def reset(self, seed=None, options=None):
        if len(self.episode_data) > 0 and self.save_episodes:
            self._save_episode()
            
        # Increment counter and reset file tracking
        self.episode_counter += 1
        self.current_episode_file = None
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = None
        self.position_size = 0
        self.position_type = None
        self.trade_duration = 0
        
        # Reset trade statistics
        self.trade_count = 0
        self.profitable_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
        self.peak_balance = self.initial_balance
        
        # Reset consecutive losses counter
        self.consecutive_losses = 0
        
        # Reset trade tracking
        self.trade_start_price = None
        self.trade_start_networth = None
        self.trade_unrealized_profit = 0.0
        
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
        # Store previous net worth for comparison
        prev_networth = self.get_networth()
        
        # Get current price
        current_price = self.df['close'].iloc[self.current_step]
        
        # Calculate unrealized profit/loss for current position
        if self.position:
            self.trade_unrealized_profit = self._calculate_unrealized_profit(current_price)
        
        # Calculate net worth before action
        pre_action_networth = self.get_networth()
        
        # Default values
        reward = 0
        profit = 0
        action_taken = None
        
        # Process actions
        if action == 0:  # Hold
            action_taken = "HOLD"
            if self.position:
                self.trade_duration += 1
                
                # Penalize holding a losing position
                if self.trade_unrealized_profit < 0:
                    # Stronger penalty for holding a losing position
                    unrealized_loss_penalty = abs(self.trade_unrealized_profit) / self.initial_balance * 0.5
                    reward -= unrealized_loss_penalty
                    
                    # Additional penalty for duration of losing trade
                    duration_penalty = 0.001 * self.trade_duration
                    reward -= duration_penalty
                else:
                    # Small reward for holding a winning position
                    unrealized_profit_reward = self.trade_unrealized_profit / self.initial_balance * 0.1
                    reward += unrealized_profit_reward
        
        elif action == 1:  # Close position
            action_taken = "CLOSE"
            if self.position:
                profit = self._close_position(current_price)
                
                # Calculate total trade performance
                trade_duration = self.trade_duration
                trade_networth_change = self.get_networth() - self.trade_start_networth
                
                # Scale reward based on profit/loss
                if profit > 0:
                    # Amplify positive trade rewards
                    reward = profit / self.initial_balance * self.positive_trade_multiplier
                    # Add bonus for quick profitable trades
                    if trade_duration < 10:
                        reward += 0.1 * reward  # 10% bonus for quick trades
                    # Reset consecutive losses
                    self.consecutive_losses = 0
                else:
                    # Amplify negative trade penalties
                    base_penalty = abs(profit) / self.initial_balance * self.negative_trade_multiplier
                    # Add consecutive loss penalty if applicable
                    consecutive_penalty = self.consecutive_loss_penalty * self.consecutive_losses
                    # Add duration penalty for long losing trades
                    duration_penalty = 0.01 * min(trade_duration / 10, 1.0)
                    reward = -(base_penalty + consecutive_penalty + duration_penalty)
                    # Increment consecutive losses
                    self.consecutive_losses += 1
                
                # Reset trade tracking
                self.trade_start_price = None
                self.trade_start_networth = None
                self.trade_unrealized_profit = 0.0
        
        elif action == 2:  # Buy
            action_taken = "BUY"
            if not self.position:
                self._open_position(current_price, 'long')
                self.trade_duration = 0
                # Small penalty for opening a position (trading fee)
                reward = -0.001
        
        elif action == 3:  # Sell
            action_taken = "SELL"
            if not self.position:
                self._open_position(current_price, 'short')
                self.trade_duration = 0
                # Small penalty for opening a position (trading fee)
                reward = -0.001
        
        # Move to next step
        self.current_step += 1
        
        # Check if we've reached the end of data
        if self.current_step >= self.max_steps - 1:
            # Force close position at the end with potential penalty
            if self.position:
                final_profit = self._close_position(current_price)
                if final_profit < 0:
                    reward = -abs(final_profit) / self.initial_balance * self.negative_trade_multiplier
            return self._next_observation(), reward, True, False, {}
        
        # Calculate net worth after action
        post_action_networth = self.get_networth()
        
        # Calculate reward based on change in net worth
        networth_change = post_action_networth - pre_action_networth
        reward_scale = 100.0  # Scale factor to make rewards more meaningful
        networth_reward = networth_change / self.initial_balance * reward_scale
        
        # Combine rewards
        if reward == 0:  # If no specific action reward was set
            reward = networth_reward
        else:
            # Add networth change component with lower weight
            reward += networth_reward * 0.5
        
        # Additional reward/penalty based on net worth change from previous step
        current_networth = self.get_networth()
        step_networth_change = (current_networth - prev_networth) / self.initial_balance
        
        # Scale networth change reward/penalty
        if step_networth_change > 0:
            reward += step_networth_change * self.positive_trade_multiplier
        else:
            reward += step_networth_change * self.negative_trade_multiplier
        
        # Penalty for holding a position too long without profit
        if self.position and self.trade_duration > 20 and self.trade_unrealized_profit <= 0:
            # Stronger penalty for long unprofitable trades
            reward -= 0.002 * (self.trade_duration - 20)
        
        done = self.current_step >= self.max_steps - 1
        self._log_step(action, reward, done, action_taken)
        
        # Update maximum drawdown
        self.max_drawdown = max(self.max_drawdown, (self.peak_balance - self.balance) / self.peak_balance)
        
        # Update peak balance
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        # Get next observation
        next_obs = self._next_observation()
        
        # Return step information
        return next_obs, reward, done, False, {}

    def _log_step(self, action, reward, done, action_taken=None):
        if not self.save_episodes or self.current_step >= len(self.df):
            return
            
        current_price = self.df['close'].iloc[self.current_step]
        networth = self.get_networth()
        
        # Use the provided action_taken if available, otherwise use the action map
        action_str = action_taken if action_taken else self.action_map[action]
        
        log_entry = {
            'datetime': self.df['datetime'].iloc[self.current_step],
            'timestamp': self.df['timestamp'].iloc[self.current_step],
            'price': current_price,
            'action': action_str,
            'position_type': self.position_type if self.position else 'None',
            'quantity': self.position_size,
            'balance': self.balance,
            'networth': networth,
            'reward': reward,
            'done': done,
            'trade_count': self.trade_count,
            'profitable_trades': self.profitable_trades,
            'win_rate': self.profitable_trades / max(1, self.trade_count),
            'total_profit': self.total_profit,
            'max_drawdown': self.max_drawdown
        }
        self.episode_data.append(log_entry)

    def _save_episode(self):
        if not self.save_episodes or not self.episode_data:
            return
            
        df = pd.DataFrame(self.episode_data)
        # Create new file at the start of each episode
        if not self.current_episode_file:
            self.current_episode_file = f"episode_{self.episode_counter:04d}.csv"
            
        # Append to current episode file
        df.to_csv(self.current_episode_file, mode='a', 
                 header=not os.path.exists(self.current_episode_file), 
                 index=False)
        self.episode_data = []

    def _open_position(self, price, position_type):
        self.position = price
        self.entry_price = price
        self.position_type = position_type
        self.position_size = (self.get_networth() * self.position_size_fraction) / price
        self.trade_duration = 0
        if position_type == 'long':
            self.balance -= price * self.position_size
        
        # Track trade start
        self.trade_start_price = price
        self.trade_start_networth = self.get_networth()
        self.trade_unrealized_profit = 0.0
        
        return self.position_size

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
        
        # Calculate total trade performance
        trade_duration = self.trade_duration
        trade_networth_change = self.get_networth() - self.trade_start_networth
        
        # Scale reward based on profit/loss
        if trade_networth_change > 0:
            # Amplify positive trade rewards
            profit = trade_networth_change / self.initial_balance * self.positive_trade_multiplier
            # Add bonus for quick profitable trades
            if trade_duration < 10:
                profit += 0.1 * profit  # 10% bonus for quick trades
            # Reset consecutive losses
            self.consecutive_losses = 0
        else:
            # Amplify negative trade penalties
            base_penalty = abs(trade_networth_change) / self.initial_balance * self.negative_trade_multiplier
            # Add consecutive loss penalty if applicable
            consecutive_penalty = self.consecutive_loss_penalty * self.consecutive_losses
            # Add duration penalty for long losing trades
            duration_penalty = 0.01 * min(trade_duration / 10, 1.0)
            profit = -(base_penalty + consecutive_penalty + duration_penalty)
            # Increment consecutive losses
            self.consecutive_losses += 1
        
        # Reset trade tracking
        self.trade_start_price = None
        self.trade_start_networth = None
        self.trade_unrealized_profit = 0.0
        
        return profit

    def _calculate_unrealized_profit(self, current_price):
        """Calculate unrealized profit/loss for current position"""
        if not self.position:
            return 0.0
            
        if self.position_type == 'long':
            return (current_price - self.entry_price) * self.position_size
        else:  # short
            return (self.entry_price - current_price) * self.position_size

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
    print("Preprocessing data...")
    btc_data = preprocess_data('BTCUSDT.data')
    print(f"Data shape: {btc_data.shape}, Columns: {btc_data.columns.tolist()}")
    
    # Create environment with debugging disabled
    trading_env = ProTraderEnv(btc_data, debug=False)
    trading_env.enable_episode_saving(True)
    
    # Wrap environment
    env = CustomActionMaskWrapper(trading_env)
    
    # Create and train agent
    print("\nInitializing agent...")
    agent = PPOTrader(env, sequence_length=20)  # Increased sequence length for better temporal patterns
    
    print("\nStarting training...")
    agent.train()