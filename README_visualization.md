# Trading Bot Learning Visualization Tools

This set of tools allows you to visualize and analyze how your reinforcement learning trading bot learns and which features have the most impact on its performance.

## Overview

The visualization tools provide several ways to understand your trading bot's learning process:

1. **Feature Importance Analysis**: Discover which features most strongly correlate with performance metrics like net worth, reward, and win rate.
2. **Learning Progress Visualization**: Track how your agent's performance improves over time.
3. **Feature Evolution**: See how the agent's response to specific features changes across training episodes.
4. **Comprehensive Dashboards**: Get a high-level overview of all important metrics in a single visualization.
5. **Detailed Feature Analysis**: Dive deep into specific features to understand their impact on trading decisions.

## Available Scripts

### 1. Visualize Learning (`visualize_learning.py`)

The core module that provides visualization functionality. You can use this directly or through the helper scripts.

```bash
python visualize_learning.py --episodes_dir training_results --data_file your_data.csv --output learning_dashboard.png
```

### 2. Example Visualization (`example_visualization.py`)

A simple script demonstrating how to use the visualization tools.

```bash
python example_visualization.py
```

### 3. Train with Visualization (`train_with_visualization.py`)

Train your trading agent while automatically generating visualizations.

```bash
python train_with_visualization.py --data_file your_data.csv --output_dir training_results --total_steps 100000
```

### 4. Analyze Feature (`analyze_feature.py`)

Perform detailed analysis of a specific feature's impact on trading performance.

```bash
python analyze_feature.py rsi_norm --episodes_dir training_results --output_dir feature_analysis
```

## Key Visualizations

### Feature Importance

Identifies which features have the strongest correlation with performance metrics:

- **Correlation Analysis**: Pearson correlation between features and performance metrics
- **Mutual Information**: Information-theoretic measure of feature importance
- **Feature Importance Plots**: Visual ranking of features by importance

### Learning Progress

Tracks how your agent's performance evolves during training:

- **Net Worth Progression**: How the agent's portfolio value changes over time
- **Reward Trends**: How rewards accumulate across episodes
- **Win Rate**: How the percentage of profitable trades changes

### Feature Analysis

Detailed analysis of specific features:

- **Feature vs. Performance**: How performance metrics change with feature values
- **Feature Evolution**: How the agent's response to a feature evolves across episodes
- **Feature Distribution**: Statistical distribution of feature values
- **Feature by Action**: How feature values relate to specific trading actions
- **Threshold Analysis**: Performance metrics at different feature thresholds

## Example Usage

### Basic Visualization

```python
from visualize_learning import LearningVisualizer

# Create visualizer
visualizer = LearningVisualizer(episodes_dir='training_results')

# Load episode data
visualizer.load_episodes()

# Create dashboard
visualizer.create_dashboard(output_file='dashboard.png')

# Find most important features
top_features = visualizer.correlate_with_performance('networth', 10)
print(top_features)

# Analyze a specific feature
visualizer.plot_feature_vs_performance('rsi_norm', 'networth')
```

### Integrating with Training

To integrate visualization with your training process, use the `train_with_visualization.py` script or modify your training loop to periodically generate visualizations.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Tips for Effective Analysis

1. **Focus on normalized features**: Features ending with `_norm` are typically more informative for correlation analysis.
2. **Compare multiple metrics**: Look at correlations with different performance metrics (net worth, reward, win rate).
3. **Examine feature evolution**: The most important features may change as training progresses.
4. **Look for action thresholds**: Identify feature values that consistently trigger specific actions.
5. **Consider feature interactions**: Some features may be more important in combination than individually.

## Troubleshooting

- **No episode files found**: Make sure you've run training with episode saving enabled.
- **Feature not found**: Check the available features by running `analyze_feature.py` with a non-existent feature name.
- **Visualization errors**: Ensure you have all required dependencies installed. 