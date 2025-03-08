import os
import argparse
import pandas as pd
import numpy as np
# Set matplotlib backend to 'Agg' to avoid Tcl/Tk dependency
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from visualize_learning import LearningVisualizer

def analyze_feature(feature_name, episodes_dir='.', output_dir='feature_analysis'):
    """
    Perform detailed analysis of a specific feature's impact on trading performance
    
    Args:
        feature_name: Name of the feature to analyze
        episodes_dir: Directory containing episode data
        output_dir: Directory to save analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = LearningVisualizer(episodes_dir=episodes_dir)
    
    try:
        # Load episode data
        visualizer.load_episodes()
        
        # Get episode data
        episode_data = visualizer.episode_data
        
        # Check if feature exists in data
        if feature_name not in episode_data.columns:
            print(f"Feature '{feature_name}' not found in episode data.")
            print("Available features:")
            print("\n".join(sorted([col for col in episode_data.columns 
                                   if col not in ['episode', 'timestamp', 'datetime']])))
            return
        
        print(f"Analyzing feature: {feature_name}")
        
        # 1. Basic statistics
        feature_stats = episode_data[feature_name].describe()
        print("\nFeature Statistics:")
        print(feature_stats)
        
        # 2. Correlation with performance metrics
        performance_metrics = ['networth', 'reward', 'win_rate', 'total_profit']
        available_metrics = [m for m in performance_metrics if m in episode_data.columns]
        
        correlations = {}
        for metric in available_metrics:
            corr = episode_data[feature_name].corr(episode_data[metric])
            correlations[metric] = corr
            
        print("\nCorrelations with Performance Metrics:")
        for metric, corr in correlations.items():
            print(f"{metric}: {corr:.4f}")
        
        # 3. Feature vs Performance Visualization
        for metric in available_metrics:
            fig = visualizer.plot_feature_vs_performance(
                feature=feature_name,
                performance_metric=metric,
                rolling_window=100
            )
            output_file = os.path.join(output_dir, f'{feature_name}_vs_{metric}.png')
            fig.savefig(output_file)
            print(f"Saved {output_file}")
        
        # 4. Feature Evolution Across Episodes
        fig = visualizer.plot_feature_evolution(
            feature=feature_name,
            n_episodes=5
        )
        output_file = os.path.join(output_dir, f'{feature_name}_evolution.png')
        plt.savefig(output_file)
        print(f"Saved {output_file}")
        
        # 5. Feature Distribution
        plt.figure(figsize=(12, 6))
        
        # Plot histogram
        plt.subplot(1, 2, 1)
        sns.histplot(episode_data[feature_name], kde=True)
        plt.title(f'Distribution of {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Frequency')
        
        # Plot boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=episode_data[feature_name])
        plt.title(f'Boxplot of {feature_name}')
        plt.ylabel(feature_name)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{feature_name}_distribution.png')
        plt.savefig(output_file)
        print(f"Saved {output_file}")
        
        # 6. Feature vs Actions
        if 'action' in episode_data.columns:
            plt.figure(figsize=(14, 8))
            
            # Box plot of feature by action
            sns.boxplot(x='action', y=feature_name, data=episode_data)
            plt.title(f'Distribution of {feature_name} by Action')
            plt.xlabel('Action')
            plt.ylabel(feature_name)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            output_file = os.path.join(output_dir, f'{feature_name}_by_action.png')
            plt.savefig(output_file)
            print(f"Saved {output_file}")
            
            # Calculate average feature value for each action
            action_means = episode_data.groupby('action')[feature_name].mean().sort_values()
            print("\nAverage Feature Value by Action:")
            for action, mean_value in action_means.items():
                print(f"{action}: {mean_value:.4f}")
        
        # 7. Feature Thresholds Analysis
        # Analyze how different thresholds of the feature affect performance
        percentiles = [10, 25, 50, 75, 90]
        thresholds = np.percentile(episode_data[feature_name].dropna(), percentiles)
        
        print("\nFeature Threshold Analysis:")
        print(f"{'Percentile':>10} {'Threshold':>12} {'Avg Reward':>12} {'Win Rate':>12} {'Networth':>12}")
        
        for p, threshold in zip(percentiles, thresholds):
            # Above threshold
            above_threshold = episode_data[episode_data[feature_name] >= threshold]
            if len(above_threshold) > 0:
                avg_reward_above = above_threshold['reward'].mean()
                win_rate_above = above_threshold['win_rate'].mean() if 'win_rate' in above_threshold.columns else np.nan
                networth_above = above_threshold['networth'].mean()
                
                print(f"{p:>10}% {threshold:>12.4f} {avg_reward_above:>12.4f} {win_rate_above:>12.4f} {networth_above:>12.4f}")
        
        print("\nFeature analysis complete!")
        
    except Exception as e:
        print(f"Error analyzing feature: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze a specific feature in detail')
    parser.add_argument('feature', type=str, help='Name of the feature to analyze')
    parser.add_argument('--episodes_dir', type=str, default='.',
                       help='Directory containing episode data')
    parser.add_argument('--output_dir', type=str, default='feature_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    analyze_feature(
        feature_name=args.feature,
        episodes_dir=args.episodes_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 