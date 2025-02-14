#!/usr/bin/env python3
import requests
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def get_metrics():
    response = requests.get('http://localhost:8000/metrics')
    metrics = {}
    for line in response.text.split('\n'):
        if line and not line.startswith('#'):
            try:
                name, value = line.split(' ')
                # Basic metrics
                if name in [
                    'refinance_model_accuracy',
                    'refinance_model_precision',
                    'refinance_model_recall',
                    'refinance_model_f1',
                    'ground_truth_refinance_rate',
                    'prediction_error_rate'
                ]:
                    metrics[name] = float(value)
                # Feature drift metrics
                elif name.startswith('feature_drift{'):
                    metrics[name] = float(value)
                # Feature importance
                elif name.startswith('feature_importance{'):
                    metrics[name] = float(value)
                # Prediction distribution
                elif name.startswith('prediction_distribution{'):
                    metrics[name] = float(value)
            except:
                continue
    return metrics

def plot_metrics(metrics_history):
    latest_metrics = metrics_history[-1]
    
    # Create subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Performance metrics over time
    ax1 = plt.subplot(221)
    df = pd.DataFrame(metrics_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    metrics_to_plot = ['refinance_model_accuracy', 'refinance_model_precision', 
                      'refinance_model_recall', 'refinance_model_f1']
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            ax1.plot(df['timestamp'], df[metric], label=metric.replace('refinance_model_', ''))
    
    ax1.set_title('Performance Metrics Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Feature Drift Metrics
    ax2 = plt.subplot(222)
    drift_metrics = {k: v for k, v in latest_metrics.items() if k.startswith('feature_drift')}
    if drift_metrics:
        features = list(set([k.split('feature="')[1].split('"')[0] for k in drift_metrics.keys()]))
        drift_types = ['ks_statistic', 'jensen_shannon_div', 'psi']
        
        x = np.arange(len(features))
        width = 0.25
        
        for i, drift_type in enumerate(drift_types):
            values = [drift_metrics.get(f'feature_drift{{feature="{f}",metric_type="{drift_type}}"}}', 0) 
                     for f in features]
            ax2.bar(x + i*width, values, width, label=drift_type)
        
        ax2.set_title('Feature Drift Metrics')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(features, rotation=45)
        ax2.legend()
    
    # Plot 3: Feature Importance
    ax3 = plt.subplot(223)
    importance_metrics = {k: v for k, v in latest_metrics.items() if k.startswith('feature_importance')}
    if importance_metrics:
        features = [k.split('feature="')[1].split('"')[0] for k in importance_metrics.keys()]
        values = [importance_metrics[k] for k in importance_metrics.keys()]
        ax3.bar(features, values)
        ax3.set_title('Feature Importance')
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Prediction Distribution
    ax4 = plt.subplot(224)
    dist_metrics = {k: v for k, v in latest_metrics.items() if k.startswith('prediction_distribution')}
    if dist_metrics:
        buckets = [k.split('bucket="')[1].split('"')[0] for k in dist_metrics.keys()]
        values = [dist_metrics[k] for k in dist_metrics.keys()]
        ax4.bar(buckets, values)
        ax4.set_title('Prediction Probability Distribution')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_metrics.png')
    print("\nMetrics plot saved as 'model_metrics.png'")

def main():
    print("Monitoring model metrics (press Ctrl+C to stop)...")
    metrics_history = []
    
    try:
        while True:
            metrics = get_metrics()
            metrics['timestamp'] = datetime.now()
            metrics_history.append(metrics)
            
            # Clear screen and show current metrics
            print("\033[H\033[J")  # Clear screen
            print("Current Model Metrics:")
            metrics_table = [[k.replace('refinance_model_', '').title(), f"{v:.3f}"] 
                           for k, v in metrics.items() if k != 'timestamp']
            print(tabulate(metrics_table, headers=['Metric', 'Value'], tablefmt='grid'))
            
            # Plot metrics if we have enough data points
            if len(metrics_history) > 1:
                plot_metrics(metrics_history)
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\nStopping metrics monitor...")
        if metrics_history:
            plot_metrics(metrics_history)

if __name__ == "__main__":
    main()
