import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec

class LearningVisualizer:
    def __init__(self, episodes_dir='.', feature_names=None):
        """
        Initialize the visualizer with episode data directory
        
        Args:
            episodes_dir: Directory containing episode CSV files
            feature_names: List of feature names to analyze (if None, will use all available)
        """
        self.episodes_dir = episodes_dir
        self.feature_names = feature_names
        self.episode_files = []
        self.episode_data = None
        self.feature_importance = None
        
    def load_episodes(self, pattern="episode_*.csv"):
        """Load all episode data from CSV files"""
        self.episode_files = sorted(glob.glob(os.path.join(self.episodes_dir, pattern)))
        
        if not self.episode_files:
            raise ValueError(f"No episode files found matching pattern {pattern} in {self.episodes_dir}")
            
        print(f"Found {len(self.episode_files)} episode files")
        
        # Load all episode data into a single DataFrame
        all_episodes = []
        for file in self.episode_files:
            try:
                df = pd.read_csv(file)
                # Add episode number based on filename
                episode_num = int(os.path.basename(file).split('_')[1].split('.')[0])
                df['episode'] = episode_num
                all_episodes.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        if not all_episodes:
            raise ValueError("No valid episode data could be loaded")
            
        self.episode_data = pd.concat(all_episodes, ignore_index=True)
        print(f"Loaded {len(self.episode_data)} total steps across {len(all_episodes)} episodes")
        
        return self.episode_data
    
    def load_feature_data(self, data_file):
        """Load the original feature data used for training"""
        self.feature_data = pd.read_csv(data_file)
        return self.feature_data
        
    def correlate_with_performance(self, performance_metric='networth', top_n=10):
        """
        Calculate correlation between features and performance metrics
        
        Args:
            performance_metric: Metric to correlate with (e.g., 'networth', 'reward')
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.episode_data is None:
            raise ValueError("No episode data loaded. Call load_episodes() first.")
            
        # Merge episode data with feature data based on timestamp
        if hasattr(self, 'feature_data'):
            merged_data = pd.merge(
                self.episode_data,
                self.feature_data,
                on='timestamp',
                how='left'
            )
        else:
            merged_data = self.episode_data
            
        # Get all feature columns (normalized versions)
        if self.feature_names is None:
            # Try to identify feature columns (those ending with _norm)
            self.feature_names = [col for col in merged_data.columns if col.endswith('_norm')]
            
            if not self.feature_names:
                # If no _norm columns, use any numeric columns except the performance metrics
                exclude_cols = ['episode', 'timestamp', 'datetime', 'networth', 'balance', 
                               'reward', 'done', 'trade_count', 'profitable_trades', 
                               'win_rate', 'total_profit', 'max_drawdown']
                self.feature_names = [col for col in merged_data.select_dtypes(include=np.number).columns 
                                     if col not in exclude_cols]
        
        # Calculate correlation with performance metric
        correlations = {}
        for feature in self.feature_names:
            if feature in merged_data.columns:
                # Calculate Pearson correlation
                corr = merged_data[feature].corr(merged_data[performance_metric])
                correlations[feature] = corr
        
        # Convert to DataFrame and sort
        self.feature_importance = pd.DataFrame({
            'feature': list(correlations.keys()),
            'correlation': list(correlations.values())
        }).sort_values('correlation', ascending=False, key=abs)
        
        return self.feature_importance.head(top_n)
    
    def calculate_mutual_information(self, target='action', top_n=10):
        """
        Calculate mutual information between features and target variable
        
        Args:
            target: Target variable (e.g., 'action', 'reward')
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores based on mutual information
        """
        if self.episode_data is None:
            raise ValueError("No episode data loaded. Call load_episodes() first.")
            
        # Merge episode data with feature data based on timestamp
        if hasattr(self, 'feature_data'):
            merged_data = pd.merge(
                self.episode_data,
                self.feature_data,
                on='timestamp',
                how='left'
            )
        else:
            merged_data = self.episode_data
            
        # Get all feature columns
        if self.feature_names is None:
            # Try to identify feature columns (those ending with _norm)
            self.feature_names = [col for col in merged_data.columns if col.endswith('_norm')]
            
            if not self.feature_names:
                # If no _norm columns, use any numeric columns except the performance metrics
                exclude_cols = ['episode', 'timestamp', 'datetime', 'networth', 'balance', 
                               'reward', 'done', 'trade_count', 'profitable_trades', 
                               'win_rate', 'total_profit', 'max_drawdown']
                self.feature_names = [col for col in merged_data.select_dtypes(include=np.number).columns 
                                     if col not in exclude_cols]
        
        # Prepare feature matrix
        X = merged_data[self.feature_names].fillna(0)
        
        # For categorical target, convert to numeric
        if target == 'action' and merged_data[target].dtype == 'object':
            # Map action strings to integers
            action_map = {action: i for i, action in enumerate(merged_data[target].unique())}
            y = merged_data[target].map(action_map)
        else:
            y = merged_data[target]
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y)
        
        # Create DataFrame with results
        mi_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        return mi_df.head(top_n)
    
    def plot_feature_importance(self, method='correlation', target='networth', top_n=10):
        """
        Plot feature importance
        
        Args:
            method: 'correlation' or 'mutual_info'
            target: Target variable for correlation/mutual info
            top_n: Number of top features to display
        """
        plt.figure(figsize=(12, 8))
        
        if method == 'correlation':
            importance_df = self.correlate_with_performance(target, top_n)
            title = f"Feature Correlation with {target.capitalize()}"
            color_map = 'RdBu_r'
        else:  # mutual_info
            importance_df = self.calculate_mutual_information(target, top_n)
            title = f"Feature Importance (Mutual Information) for {target.capitalize()}"
            color_map = 'viridis'
        
        # Plot horizontal bar chart
        ax = sns.barplot(
            x='correlation' if method == 'correlation' else 'importance',
            y='feature',
            data=importance_df,
            palette=color_map if method == 'correlation' else None
        )
        
        plt.title(title, fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        return ax
    
    def plot_feature_vs_performance(self, feature, performance_metric='networth', 
                                   rolling_window=100, figsize=(14, 8)):
        """
        Plot a specific feature against performance over time
        
        Args:
            feature: Feature name to plot
            performance_metric: Performance metric to plot against
            rolling_window: Window size for rolling average
            figsize: Figure size
        """
        if self.episode_data is None:
            raise ValueError("No episode data loaded. Call load_episodes() first.")
            
        # Merge episode data with feature data based on timestamp if needed
        if hasattr(self, 'feature_data') and feature not in self.episode_data.columns:
            merged_data = pd.merge(
                self.episode_data,
                self.feature_data,
                on='timestamp',
                how='left'
            )
        else:
            merged_data = self.episode_data
            
        if feature not in merged_data.columns:
            raise ValueError(f"Feature '{feature}' not found in data")
            
        # Create figure with two subplots sharing x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot performance metric
        ax1.plot(merged_data[performance_metric], 'b-', alpha=0.3, label=f'Raw {performance_metric}')
        ax1.plot(merged_data[performance_metric].rolling(rolling_window).mean(), 'b-', 
                linewidth=2, label=f'{rolling_window}-period MA')
        ax1.set_title(f'{performance_metric.capitalize()} vs {feature}', fontsize=16)
        ax1.set_ylabel(performance_metric.capitalize(), fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot feature
        ax2.plot(merged_data[feature], 'g-', alpha=0.7, label=feature)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel(feature, fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_learning_progress(self, metrics=None, rolling_window=10, figsize=(14, 10)):
        """
        Plot learning progress over episodes
        
        Args:
            metrics: List of metrics to plot (default: ['networth', 'reward', 'win_rate'])
            rolling_window: Window size for rolling average
            figsize: Figure size
        """
        if self.episode_data is None:
            raise ValueError("No episode data loaded. Call load_episodes() first.")
            
        if metrics is None:
            metrics = ['networth', 'reward', 'win_rate']
            
        # Group by episode and calculate end-of-episode metrics
        episode_metrics = self.episode_data.groupby('episode').agg({
            'networth': 'last',
            'reward': 'sum',
            'win_rate': 'last',
            'total_profit': 'last',
            'max_drawdown': 'last'
        }).reset_index()
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
        if len(metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            if metric not in episode_metrics.columns:
                print(f"Warning: Metric '{metric}' not found in data")
                continue
                
            # Plot raw values and rolling average
            axes[i].plot(episode_metrics['episode'], episode_metrics[metric], 
                        'b-', alpha=0.3, label=f'Raw {metric}')
            
            # Plot rolling average
            rolling_avg = episode_metrics[metric].rolling(rolling_window).mean()
            axes[i].plot(episode_metrics['episode'], rolling_avg, 
                        'b-', linewidth=2, label=f'{rolling_window}-episode MA')
            
            axes[i].set_ylabel(metric.capitalize(), fontsize=12)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
        axes[0].set_title('Learning Progress by Episode', fontsize=16)
        axes[-1].set_xlabel('Episode', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_heatmap(self, top_n=15, target='networth'):
        """
        Plot heatmap of feature correlations with performance metrics
        
        Args:
            top_n: Number of top features to include
            target: Target performance metric
        """
        if self.episode_data is None:
            raise ValueError("No episode data loaded. Call load_episodes() first.")
            
        # Get top features by correlation with target
        top_features = self.correlate_with_performance(target, top_n)['feature'].tolist()
        
        # Merge episode data with feature data based on timestamp if needed
        if hasattr(self, 'feature_data'):
            merged_data = pd.merge(
                self.episode_data,
                self.feature_data,
                on='timestamp',
                how='left'
            )
        else:
            merged_data = self.episode_data
        
        # Calculate correlation matrix
        performance_metrics = ['networth', 'reward', 'win_rate', 'total_profit']
        available_metrics = [m for m in performance_metrics if m in merged_data.columns]
        
        # Combine top features and performance metrics
        columns_to_correlate = top_features + available_metrics
        
        # Filter to only include columns that exist in the data
        columns_to_correlate = [col for col in columns_to_correlate if col in merged_data.columns]
        
        # Calculate correlation matrix
        corr_matrix = merged_data[columns_to_correlate].corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .5})
        
        plt.title('Feature Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_evolution(self, feature, n_episodes=5, figsize=(14, 8)):
        """
        Plot how a specific feature's impact evolves over episodes
        
        Args:
            feature: Feature name to analyze
            n_episodes: Number of episodes to sample
            figsize: Figure size
        """
        if self.episode_data is None:
            raise ValueError("No episode data loaded. Call load_episodes() first.")
            
        # Merge episode data with feature data based on timestamp if needed
        if hasattr(self, 'feature_data') and feature not in self.episode_data.columns:
            merged_data = pd.merge(
                self.episode_data,
                self.feature_data,
                on='timestamp',
                how='left'
            )
        else:
            merged_data = self.episode_data
            
        if feature not in merged_data.columns:
            raise ValueError(f"Feature '{feature}' not found in data")
            
        # Get unique episodes
        episodes = merged_data['episode'].unique()
        
        # Sample n_episodes evenly from the range
        if len(episodes) <= n_episodes:
            sample_episodes = episodes
        else:
            # Sample episodes evenly
            indices = np.linspace(0, len(episodes)-1, n_episodes, dtype=int)
            sample_episodes = episodes[indices]
            
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot feature vs reward for each sampled episode
        for episode in sample_episodes:
            episode_data = merged_data[merged_data['episode'] == episode]
            plt.scatter(episode_data[feature], episode_data['reward'], 
                       alpha=0.7, label=f'Episode {episode}')
            
        plt.title(f'Evolution of {feature} Impact on Reward', fontsize=16)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def create_dashboard(self, output_file='learning_dashboard.png'):
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            output_file: Output file path for the dashboard image
        """
        if self.episode_data is None:
            raise ValueError("No episode data loaded. Call load_episodes() first.")
            
        # Create figure with grid layout
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # 1. Learning progress plot
        ax1 = fig.add_subplot(gs[0, :])
        episode_metrics = self.episode_data.groupby('episode').agg({
            'networth': 'last',
            'reward': 'sum',
            'win_rate': 'last'
        }).reset_index()
        ax1.plot(episode_metrics['episode'], episode_metrics['networth'], 'b-', label='Net Worth')
        ax1.set_title('Learning Progress (Net Worth by Episode)', fontsize=16)
        ax1.set_ylabel('Net Worth', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature importance plot
        ax2 = fig.add_subplot(gs[1, 0])
        importance_df = self.correlate_with_performance('networth', 8)
        sns.barplot(
            x='correlation',
            y='feature',
            data=importance_df,
            palette='RdBu_r',
            ax=ax2
        )
        ax2.set_title('Feature Correlation with Net Worth', fontsize=14)
        
        # 3. Win rate progress
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(episode_metrics['episode'], episode_metrics['win_rate'], 'g-', label='Win Rate')
        ax3.set_title('Win Rate by Episode', fontsize=14)
        ax3.set_ylabel('Win Rate', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Reward progress
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(episode_metrics['episode'], episode_metrics['reward'], 'r-', label='Total Reward')
        ax4.set_title('Total Reward by Episode', fontsize=14)
        ax4.set_ylabel('Reward', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # 5. Feature correlation heatmap (small version)
        ax5 = fig.add_subplot(gs[2, :2])
        
        # Get top features by correlation with networth
        top_features = importance_df['feature'].tolist()[:5]
        
        # Calculate correlation matrix for top features and performance metrics
        performance_metrics = ['networth', 'reward', 'win_rate']
        available_metrics = [m for m in performance_metrics if m in self.episode_data.columns]
        
        # Combine top features and performance metrics
        columns_to_correlate = top_features + available_metrics
        
        # Filter to only include columns that exist in the data
        columns_to_correlate = [col for col in columns_to_correlate 
                               if col in self.episode_data.columns]
        
        # Calculate correlation matrix
        corr_matrix = self.episode_data[columns_to_correlate].corr()
        
        # Plot heatmap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr_matrix, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, annot=True, fmt=".2f", 
                   cbar_kws={"shrink": .5}, ax=ax5)
        
        ax5.set_title('Feature-Performance Correlation Matrix', fontsize=14)
        
        # 6. Action distribution
        ax6 = fig.add_subplot(gs[2, 2])
        action_counts = self.episode_data['action'].value_counts()
        # Convert index to list for pie chart labels
        ax6.pie(action_counts.values, labels=action_counts.index.tolist(), autopct='%1.1f%%', 
               shadow=True, startangle=90)
        ax6.set_title('Action Distribution', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        
        print(f"Dashboard saved to {output_file}")
        return fig


def main():
    """Example usage of the LearningVisualizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize RL trading bot learning')
    parser.add_argument('--episodes_dir', type=str, default='.', 
                       help='Directory containing episode CSV files')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Original data file with features (optional)')
    parser.add_argument('--output', type=str, default='learning_dashboard.png',
                       help='Output file for dashboard')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = LearningVisualizer(episodes_dir=args.episodes_dir)
    
    # Load episode data
    visualizer.load_episodes()
    
    # Load feature data if provided
    if args.data_file:
        visualizer.load_feature_data(args.data_file)
    
    # Create dashboard
    visualizer.create_dashboard(output_file=args.output)
    
    # Show top features by correlation with networth
    top_features = visualizer.correlate_with_performance('networth', 10)
    print("\nTop features by correlation with net worth:")
    print(top_features)
    
    # Show top features by mutual information with action
    top_mi_features = visualizer.calculate_mutual_information('action', 10)
    print("\nTop features by mutual information with action:")
    print(top_mi_features)


if __name__ == "__main__":
    main() 