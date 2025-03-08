import os
import argparse
import pandas as pd
# Set matplotlib backend to 'Agg' to avoid Tcl/Tk dependency
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from proTradeRL import PPOTrader, ProTraderEnv, CustomActionMaskWrapper, preprocess_data
from visualize_learning import LearningVisualizer

def train_with_visualization(data_file, output_dir='training_results', 
                            total_steps=100_000, visualize_every=10):
    """
    Train the PPO trading agent with periodic visualization of feature importance
    
    Args:
        data_file: Path to the data file for training
        output_dir: Directory to save results and visualizations
        total_steps: Total number of training steps
        visualize_every: Visualize after this many episodes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess data
    print(f"Preprocessing data from {data_file}...")
    df = preprocess_data(data_file)
    
    # Create environment
    print("Creating trading environment...")
    env = ProTraderEnv(df, initial_balance=10000, debug=False)
    env = CustomActionMaskWrapper(env)
    
    # Create agent with improved parameters for technical indicator learning
    print("Initializing PPO agent with optimized parameters for technical indicators...")
    agent = PPOTrader(
        env=env,
        hidden_size=512,  # Larger network to capture complex patterns
        policy_lr=5e-5,   # Lower learning rate for better convergence
        gamma=0.99,       # High discount factor for long-term rewards
        clip_epsilon=0.2,
        batch_size=2048,  # Larger batch size for more stable learning
        ent_coef=0.02,    # Slightly higher entropy for better exploration
        gae_lambda=0.95,
        sequence_length=20  # Longer sequence to capture indicator patterns
    )
    
    # Initialize visualizer
    visualizer = LearningVisualizer(episodes_dir=output_dir)
    
    # Track episodes for visualization
    episode_counter = 0
    last_visualized = -1
    
    # Custom training loop with visualization
    print(f"Starting training for {total_steps} steps...")
    
    # Use the agent's train method but with custom visualization
    pbar = agent.train(total_steps)
    
    # After training, create comprehensive visualizations
    print("\nTraining complete! Creating visualizations...")
    
    try:
        # Load all episode data
        visualizer.load_episodes()
        
        # Create dashboard
        dashboard_file = os.path.join(output_dir, 'final_dashboard.png')
        visualizer.create_dashboard(output_file=dashboard_file)
        
        # Get top features by correlation with networth
        top_features = visualizer.correlate_with_performance('networth', 10)
        print("\nTop features by correlation with net worth:")
        print(top_features)
        
        # Get features most influencing action choice
        action_features = visualizer.calculate_mutual_information('action', 10)
        print("\nTop features influencing action choice:")
        print(action_features)
        
        # Create feature importance plot for action choice
        action_importance_file = os.path.join(output_dir, 'action_feature_importance.png')
        visualizer.plot_feature_importance(method='mutual_info', target='action')
        plt.savefig(action_importance_file)
        
        # Specifically analyze RSI features
        rsi_features = [f for f in visualizer.episode_data.columns if 'rsi' in f.lower()]
        print("\nAnalyzing RSI features:")
        print(rsi_features)
        
        for feature in rsi_features:
            if feature in visualizer.episode_data.columns:
                print(f"\nAnalyzing RSI feature: {feature}")
                
                # Plot feature vs performance
                feature_file = os.path.join(output_dir, f'feature_{feature}_analysis.png')
                try:
                    fig = visualizer.plot_feature_vs_performance(feature, 'networth')
                    fig.savefig(feature_file)
                    print(f"Saved {feature_file}")
                except Exception as e:
                    print(f"Error plotting {feature}: {e}")
        
        # Visualize top features
        if not top_features.empty:
            for i, (_, row) in enumerate(top_features.iterrows()):
                if i >= 3:  # Only visualize top 3 features
                    break
                    
                feature = row['feature']
                correlation = row['correlation']
                
                print(f"\nAnalyzing feature: {feature} (correlation: {correlation:.4f})")
                
                # Plot feature vs performance
                feature_file = os.path.join(output_dir, f'feature_{feature}_analysis.png')
                fig = visualizer.plot_feature_vs_performance(feature, 'networth')
                fig.savefig(feature_file)
                
                # Plot feature evolution
                evolution_file = os.path.join(output_dir, f'feature_{feature}_evolution.png')
                fig = visualizer.plot_feature_evolution(feature, n_episodes=5)
                plt.savefig(evolution_file)
                
        # Create feature importance plot
        importance_file = os.path.join(output_dir, 'feature_importance.png')
        visualizer.plot_feature_importance(method='correlation', target='networth')
        plt.savefig(importance_file)
        
        # Create feature correlation heatmap
        heatmap_file = os.path.join(output_dir, 'feature_correlation.png')
        visualizer.plot_feature_heatmap(top_n=10)
        plt.savefig(heatmap_file)
        
        print(f"\nAll visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    return agent

def main():
    parser = argparse.ArgumentParser(description='Train PPO trading agent with visualization')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to the data file for training')
    parser.add_argument('--output_dir', type=str, default='training_results',
                       help='Directory to save results and visualizations')
    parser.add_argument('--total_steps', type=int, default=100_000,
                       help='Total number of training steps')
    parser.add_argument('--visualize_every', type=int, default=10,
                       help='Visualize after this many episodes')
    
    args = parser.parse_args()
    
    train_with_visualization(
        data_file=args.data_file,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        visualize_every=args.visualize_every
    )

if __name__ == "__main__":
    main() 