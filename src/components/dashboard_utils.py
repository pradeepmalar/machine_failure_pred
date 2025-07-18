# src/dashboard_utils.py

import matplotlib.pyplot as plt

def plot_performance_metrics(metrics_dict):
    """
    Plots a bar chart of key performance metrics.
    metrics_dict: Dictionary with keys like 'accuracy', 'precision', etc.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    values = [metrics_dict[m] for m in metrics]
    colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF']

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=colors)
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def show_business_impact():
    """
    Prints and/or displays summarized business impact values.
    Edit values as appropriate for your context.
    """
    data = [
        ("Prevented Failures", "$2,050,000"),
        ("Maintenance Costs",  "$275,000"),
        ("False Alarm Costs",  "$14,000"),
        ("Missed Failure Costs", "$1,000,000"),
        ("Net Savings",        "$761,000"),
        ("Annual Savings",     "$3,805,000"),
        ("ROI",                "7510.0%")
    ]
    print("\nBusiness Impact Summary:")
    print("-" * 40)
    for label, value in data:
        print(f"{label:<25} : {value}")
    print("-" * 40)
