import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load baseline and distillation results."""
    baseline_df = pd.read_csv('baseline_results.csv')
    distillation_df = pd.read_csv('distillation_results.csv')
    
    print(f"Loaded {len(baseline_df)} baseline results")
    print(f"Loaded {len(distillation_df)} distillation results")
    
    return baseline_df, distillation_df

def merge_baseline_distillation(baseline_df, distillation_df):
    """Merge baseline and distillation results for comparison."""
    # Rename baseline model column to match distillation
    baseline_renamed = baseline_df.rename(columns={'baseline_model': 'student_model'})
    baseline_renamed['method'] = 'baseline'
    
    # Add method column to distillation
    distillation_df = distillation_df.copy()
    distillation_df['method'] = 'distillation'
    
    # Combine datasets
    combined_df = pd.concat([
        baseline_renamed[['dataset', 'numshot', 'student_model', 'mean_auc', 'mean_complexity', 'method']],
        distillation_df[['dataset', 'numshot', 'student_model', 'mean_auc', 'mean_complexity', 'method', 'parent_model']]
    ], ignore_index=True)
    
    return combined_df

def create_comparison_plots(baseline_df, distillation_df):
    """Create comprehensive comparison plots."""
    
    # Create output directory
    output_dir = Path('visualization_results')
    output_dir.mkdir(exist_ok=True)
    
    # Get unique parent models
    parent_models = distillation_df['parent_model'].unique()
    student_models = distillation_df['student_model'].unique()
    
    for parent_model in parent_models:
        print(f"\nCreating plots for parent model: {parent_model}")
        
        # Filter distillation results for this parent model
        parent_distill = distillation_df[distillation_df['parent_model'] == parent_model].copy()
        
        # Merge with baseline for comparison
        merged_df = merge_baseline_distillation(baseline_df, parent_distill)
        
        # Create plots for this parent model
        create_parent_model_plots(merged_df, parent_model, output_dir)


    # for student_model in student_models:
    #     print(f"\nCreating plots for student model: {student_model}")
        
    #     # Filter distillation results for this student model
    #     student_distill = distillation_df[distillation_df['student_model'] == student_model].copy()
        
    #     # Filter baseline results for this student model
    #     student_baseline = baseline_df[baseline_df['baseline_model'] == student_model].copy()
        
    #     # Merge distillation and baseline data
    #     merged_df = merge_baseline_distillation(student_baseline, student_distill)
        
    #     # Create plots for this student model
    #     create_student_model_plots(merged_df, student_model, output_dir)


def create_parent_model_plots(merged_df, parent_model, output_dir):
    """Create plots for a specific parent model."""
    
    # 3 horizontal subplots for each parent model
    plt.figure(figsize=(21, 6))
    
    # 1. Box plot comparison
    plt.subplot(1, 3, 1)
    sns.boxplot(data=merged_df, x='student_model', y='mean_auc', hue='method')
    plt.title(f'{parent_model}: AUC Distribution by Student Model')
    plt.xticks(rotation=15)
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(title='Method')
    
    # 2. Shot setting analysis
    plt.subplot(1, 3, 2)
    if parent_model.lower() == 'carte':
        shot_values = [8, 16, 32, 64, 128, 256]
    else:
        shot_values = [4, 8, 16, 32, 64, 128, 256]
    student_models = merged_df['student_model'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(student_models)))
    for i, student in enumerate(student_models):
        student_data = merged_df[merged_df['student_model'] == student]
        baseline_data = student_data[student_data['method'] == 'baseline']
        if len(baseline_data) > 0:
            baseline_shot = baseline_data.groupby('numshot')['mean_auc'].mean().reset_index()
            baseline_x = []
            baseline_y = []
            for _, row in baseline_shot.iterrows():
                if row['numshot'] in shot_values:
                    baseline_x.append(row['numshot'])
                    baseline_y.append(row['mean_auc'])
            if baseline_x:
                plt.plot(baseline_x, baseline_y, 
                        color=colors[i], linestyle='--', marker='o', alpha=0.7,
                        label=f'{student} (baseline)', linewidth=2)
        distill_data = student_data[student_data['method'] == 'distillation']
        if len(distill_data) > 0:
            distill_shot = distill_data.groupby('numshot')['mean_auc'].mean().reset_index()
            distill_x = []
            distill_y = []
            for _, row in distill_shot.iterrows():
                if row['numshot'] in shot_values:
                    distill_x.append(row['numshot'])
                    distill_y.append(row['mean_auc'])
            if distill_x:
                plt.plot(distill_x, distill_y, 
                        color=colors[i], linestyle='-', marker='s', alpha=0.9,
                        label=f'{student} (distillation (ours))', linewidth=2)
    plt.xticks(shot_values)
    plt.xlim(min(shot_values) * 0.8, max(shot_values) * 1.1)
    plt.title(f'{parent_model}: AUC vs Shot Setting by Student Model')
    plt.xlabel('Number of Shots')
    plt.ylabel('Mean AUC')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 3. Overall shot setting comparison (averaged across all student models)
    plt.subplot(1, 3, 3)
    shot_comparison = merged_df.groupby(['numshot', 'method'])['mean_auc'].mean().reset_index()
    baseline_shot = shot_comparison[shot_comparison['method'] == 'baseline']
    baseline_x = []
    baseline_y = []
    for _, row in baseline_shot.iterrows():
        if row['numshot'] in shot_values:
            baseline_x.append(row['numshot'])
            baseline_y.append(row['mean_auc'])
    if baseline_x:
        plt.plot(baseline_x, baseline_y, 
                color='blue', linestyle='--', marker='o', alpha=0.8,
                label='Baseline (avg)', linewidth=3, markersize=8)
    distill_shot = shot_comparison[shot_comparison['method'] == 'distillation']
    distill_x = []
    distill_y = []
    for _, row in distill_shot.iterrows():
        if row['numshot'] in shot_values:
            distill_x.append(row['numshot'])
            distill_y.append(row['mean_auc'])
    if distill_x:
        plt.plot(distill_x, distill_y, 
                color='red', linestyle='-', marker='s', alpha=0.8,
                label='Distillation (ours) (avg)', linewidth=3, markersize=8)
    plt.xticks(shot_values)
    plt.xlim(min(shot_values) * 0.8, max(shot_values) * 1.1)
    plt.title(f'{parent_model}: Average AUC vs Shot Setting')
    plt.xlabel('Number of Shots')
    plt.ylabel('Mean AUC (averaged across all models)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{parent_model}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_student_model_plots(merged_df, student_model, output_dir):
    """Create plots for a specific student model comparing different parent models."""
    
    plt.figure(figsize=(15, 10))
    
    # Get unique parent models for this student
    parent_models = merged_df['parent_model'].dropna().unique()
    
    # Create pivot table for heatmap
    pivot_data = merged_df.pivot_table(
        values='mean_auc', 
        index=['dataset', 'numshot'], 
        columns=['parent_model', 'method'], 
        aggfunc='mean'
    )
    
    # Calculate improvement (distillation - baseline) for each parent model
    improvement_data = {}
    for parent in parent_models:
        if (parent, 'distillation') in pivot_data.columns and ('baseline', 'baseline') in pivot_data.columns:
            # Compare distillation from this parent vs baseline
            baseline_col = None
            # Find baseline column (it might be indexed differently)
            for col in pivot_data.columns:
                if col[1] == 'baseline':
                    baseline_col = col
                    break
            
            if baseline_col is not None:
                improvement_data[parent] = pivot_data[(parent, 'distillation')] - pivot_data[baseline_col]
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        
        plt.subplot(2, 2, 1)
        sns.heatmap(improvement_df, annot=False, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'AUC Improvement (Distillation - Baseline)'})
        plt.title(f'{student_model}: AUC Improvement by Parent Model')
        plt.ylabel('Dataset, Shot Setting')
    
    # 2. Box plot comparison across parent models
    plt.subplot(2, 2, 2)
    # Create data for box plot
    plot_data = merged_df.copy()
    plot_data['model_method'] = plot_data.apply(
        lambda x: f"{x['parent_model']}" if x['method'] == 'distillation' else 'Baseline', axis=1
    )
    
    sns.boxplot(data=plot_data, x='model_method', y='mean_auc')
    plt.title(f'{student_model}: AUC Distribution by Parent Model')
    plt.xticks(rotation=45)
    plt.xlabel('Method')
    
    # 3. Shot setting analysis comparing parent models
    plt.subplot(2, 2, 3)
    
    # Define shot values
    shot_values = [4, 8, 16, 32, 64, 128, 256]
    
    # Plot baseline first
    baseline_data = merged_df[merged_df['method'] == 'baseline']
    if len(baseline_data) > 0:
        baseline_shot = baseline_data.groupby('numshot')['mean_auc'].mean().reset_index()
        baseline_x = []
        baseline_y = []
        for _, row in baseline_shot.iterrows():
            if row['numshot'] in shot_values:
                baseline_x.append(row['numshot'])
                baseline_y.append(row['mean_auc'])
        
        if baseline_x:
            plt.plot(baseline_x, baseline_y, 
                    color='black', linestyle='--', marker='o', alpha=0.8,
                    label='Baseline', linewidth=3, markersize=8)
    
    # Plot each parent model
    colors = plt.cm.Set1(np.linspace(0, 1, len(parent_models)))
    
    for i, parent in enumerate(parent_models):
        parent_data = merged_df[(merged_df['parent_model'] == parent) & (merged_df['method'] == 'distillation')]
        if len(parent_data) > 0:
            parent_shot = parent_data.groupby('numshot')['mean_auc'].mean().reset_index()
            parent_x = []
            parent_y = []
            for _, row in parent_shot.iterrows():
                if row['numshot'] in shot_values:
                    parent_x.append(row['numshot'])
                    parent_y.append(row['mean_auc'])
            
            if parent_x:
                plt.plot(parent_x, parent_y, 
                        color=colors[i], linestyle='-', marker='s', alpha=0.9,
                        label=f'{parent} (distillation)', linewidth=2, markersize=6)
    
    plt.xticks(shot_values)
    plt.xlim(min(shot_values) * 0.8, max(shot_values) * 1.1)
    plt.title(f'{student_model}: AUC vs Shot Setting by Parent Model')
    plt.xlabel('Number of Shots')
    plt.ylabel('Mean AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Overall comparison (averaged across all shot settings)
    plt.subplot(2, 2, 4)
    
    # Calculate average performance across all shot settings for each parent model
    overall_comparison = merged_df.groupby(['parent_model', 'method'])['mean_auc'].mean().reset_index()
    
    # Add baseline comparison
    baseline_avg = merged_df[merged_df['method'] == 'baseline']['mean_auc'].mean()
    
    # Plot parent model performances
    parent_avgs = overall_comparison[overall_comparison['method'] == 'distillation']
    
    if len(parent_avgs) > 0:
        x_pos = np.arange(len(parent_models))
        parent_values = []
        parent_labels = []
        
        for parent in parent_models:
            parent_val = parent_avgs[parent_avgs['parent_model'] == parent]['mean_auc'].values
            if len(parent_val) > 0:
                parent_values.append(parent_val[0])
                parent_labels.append(parent)
        
        # Create bar plot
        bars = plt.bar(range(len(parent_values)), parent_values, alpha=0.7, color=colors[:len(parent_values)])
        
        # Add baseline line
        plt.axhline(y=baseline_avg, color='black', linestyle='--', linewidth=2, label=f'Baseline (avg: {baseline_avg:.3f})')
        
        plt.xticks(range(len(parent_labels)), parent_labels, rotation=45)
        plt.title(f'{student_model}: Average AUC by Parent Model')
        plt.xlabel('Parent Model')
        plt.ylabel('Average AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, parent_values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{student_model}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to create all visualizations."""
    print("Loading data...")
    baseline_df, distillation_df = load_data()
    
    print("\nCreating visualizations...")
    
    # Create output directory
    output_dir = Path('visualization_results')
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison plots for each parent model
    create_comparison_plots(baseline_df, distillation_df)



if __name__ == "__main__":
    main()
