import os
import pandas as pd
import matplotlib.pyplot as plt
from visualize_learning import LearningVisualizer

def main():
    """
    Example script demonstrating how to use the LearningVisualizer
    to analyze feature importance in the trading bot's learning process.
    """
    # Create visualizer (point to directory with episode files)
    visualizer = LearningVisualizer(episodes_dir='.')
    
    # Load episode data
    try:
        visualizer.load_episodes()
    except ValueError as e:
        print(f"Error: {e}")
        print("No episode files found. Make sure to run training first.")
        return
    
    # 1. Visualize overall learning progress
    print("Generating learning progress plot...")
    progress_fig = visualizer.plot_learning_progress(
        metrics=['networth', 'reward', 'win_rate'], 
        rolling_window=5
    )
    progress_fig.savefig('learning_progress.png')
    
    # 2. Find most important features by correlation with net worth
    print("\nTop features by correlation with net worth:")
    top_features = visualizer.correlate_with_performance('networth', 10)
    print(top_features)
    
    # 3. Find most important features by mutual information with actions
    print("\nTop features by mutual information with actions:")
    top_mi_features = visualizer.calculate_mutual_information('action', 10)
    print(top_mi_features)
    
    # 4. Plot feature importance
    print("\nGenerating feature importance plot...")
    visualizer.plot_feature_importance(method='correlation', target='networth')
    plt.savefig('feature_importance_networth.png')
    
    visualizer.plot_feature_importance(method='mutual_info', target='action')
    plt.savefig('feature_importance_action.png')
    
    # 5. If we have top features, visualize their relationship with performance
    if not top_features.empty:
        top_feature = top_features.iloc[0]['feature']
        print(f"\nAnalyzing top feature: {top_feature}")
        
        # Plot feature vs performance over time
        feature_fig = visualizer.plot_feature_vs_performance(
            feature=top_feature, 
            performance_metric='networth'
        )
        feature_fig.savefig(f'feature_analysis_{top_feature}.png')
        
        # Plot feature evolution across episodes
        evolution_fig = visualizer.plot_feature_evolution(
            feature=top_feature, 
            n_episodes=5
        )
        plt.savefig(f'feature_evolution_{top_feature}.png')
    
    # 6. Create comprehensive dashboard
    print("\nGenerating comprehensive dashboard...")
    visualizer.create_dashboard(output_file='learning_dashboard.png')
    
    print("\nVisualization complete! Check the generated PNG files.")
    print("To analyze specific features, you can use the following methods:")
    print("  - visualizer.plot_feature_vs_performance(feature_name)")
    print("  - visualizer.plot_feature_evolution(feature_name)")
    print("  - visualizer.plot_feature_heatmap()")

if __name__ == "__main__":
    main() 