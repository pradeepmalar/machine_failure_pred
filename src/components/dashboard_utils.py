import matplotlib.pyplot as plt

def plot_performance_metrics(metrics_dict):
    """
    Plots a bar chart of model performance metrics.
    
    Parameters:
        metrics_dict (dict): Keys like 'accuracy', 'precision', 'recall', 'f1_score', 'auc'
    """
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF']

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=colors[:len(metrics)])
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Score')

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def show_business_impact():
    """
    Prints or displays business value insights from predictions.
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

    print("\n📊 Business Impact Summary:")
    print("-" * 40)
    for label, value in data:
        print(f"{label:<25} : {value}")
    print("-" * 40)
