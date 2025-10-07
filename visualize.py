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
    
    # Shorten model names for better visualization
    model_name_map = {
        'logistic_rule_regression': 'log. rule regr.',
        'logistic_regression': 'log. reg.'
    }
    combined_df['student_model'] = combined_df['student_model'].replace(model_name_map)
    
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


def create_parent_model_plots(merged_df, parent_model, output_dir):
    """Create plots for a specific parent model."""
    
    # 3 horizontal subplots for each parent model
    plt.figure(figsize=(21, 6))
    
    # Add main title for the parent model
    # Convert parent model names to proper capitalization
    model_name_map = {
        'tabllm': 'TabLLM',
        'carte': 'CARTE', 
        'tabpfn': 'TabPFN',
        'tabm': 'TabM'
    }
    display_name = model_name_map.get(parent_model.lower(), parent_model)
    plt.suptitle(f'{display_name}', fontsize=32, fontweight='bold', y=0.95)
    
    # 1. Box plot comparison
    plt.subplot(1, 3, 1)
    # Define student model order alphabetically
    student_model_order = ['decision_tree', 'log. reg.', 'log. rule regr.', 'ttnet', 'xgboost']
    sns.boxplot(data=merged_df, x='student_model', y='mean_auc', hue='method', order=student_model_order)
    plt.xticks(rotation=15, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('', fontsize=16)
    plt.ylabel('', fontsize=16)
    plt.legend(title='Method', fontsize=14, title_fontsize=16)
    # plt.text(0.5, -0.25, 'AUC Distribution by Student Model', 
    #          transform=plt.gca().transAxes, ha='center', va='top', 
    #          fontsize=18, fontweight='bold')
    
    # 2. Shot setting analysis
    plt.subplot(1, 3, 2)
    if parent_model.lower() == 'carte':
        shot_values = [8, 16, 32, 64, 128, 256]
        xtick_values = [8, 32, 128]
    else:
        shot_values = [4, 8, 16, 32, 64, 128, 256]
        xtick_values = [4, 16, 64, 256]
    
    # Define student model order alphabetically
    student_model_order = ['decision_tree', 'log. reg.', 'log. rule regr.', 'ttnet', 'xgboost']
    # Filter to only include models that exist in the data
    student_models = [model for model in student_model_order if model in merged_df['student_model'].unique()]
    colors = plt.cm.Set1(np.linspace(0, 1, len(student_models)))
    
    # First, plot all lines without labels for individual student models
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
                        linewidth=2)
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
                        linewidth=2)
    
    # Create custom legend entries
    legend_elements = []
    # Add student model colors
    for i, student in enumerate(student_models):
        legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=3, label=student))
    # Add line style explanations
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Baseline'))
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Distillation (ours)'))
    
    plt.legend(handles=legend_elements, loc='best', fontsize=14)
    # Set minor ticks for all shot values but only label the specified ones
    plt.gca().set_xticks(shot_values, minor=True)
    plt.gca().set_xticks(xtick_values, minor=False)
    plt.gca().tick_params(axis='x', which='minor', length=8, width=2)
    plt.gca().tick_params(axis='x', which='major', length=4, width=1)
    plt.xticks(xtick_values, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(min(shot_values) * 0.8, max(shot_values) * 1.1)
    plt.xlabel('Number of Shots', fontsize=16)
    plt.ylabel('Mean AUC', fontsize=16)
    plt.grid(True, alpha=0.3)
    # plt.text(0.5, -0.25, 'AUC vs Shot Setting by Student Model', 
    #          transform=plt.gca().transAxes, ha='center', va='top', 
    #          fontsize=18, fontweight='bold')
    
    # 3. Overall shot setting comparison (averaged across all student models)
    plt.subplot(1, 3, 3)
    shot_comparison = merged_df.groupby(['numshot', 'method'])['mean_auc'].mean().reset_index()
    shot_comparison.columns = ['numshot', 'method', 'mean_auc']
    
    baseline_shot = shot_comparison[shot_comparison['method'] == 'baseline']
    baseline_x = []
    baseline_y = []
    for _, row in baseline_shot.iterrows():
        if row['numshot'] in shot_values:
            baseline_x.append(row['numshot'])
            baseline_y.append(row['mean_auc'])
    if baseline_x:
        baseline_y = np.array(baseline_y)
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
        distill_y = np.array(distill_y)
        plt.plot(distill_x, distill_y, 
                color='red', linestyle='-', marker='s', alpha=0.8,
                label='Distillation (ours)', linewidth=3, markersize=8)
    # Set minor ticks for all shot values but only label the specified ones
    plt.gca().set_xticks(shot_values, minor=True)
    plt.gca().set_xticks(xtick_values, minor=False)
    plt.gca().tick_params(axis='x', which='minor', length=8, width=2)
    plt.gca().tick_params(axis='x', which='major', length=4, width=1)
    plt.xticks(xtick_values, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(min(shot_values) * 0.8, max(shot_values) * 1.1)
    plt.xlabel('Number of Shots', fontsize=16)
    plt.ylabel('Mean AUC (averaged across all models)', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    # plt.text(0.5, -0.25, 'Average AUC vs Shot Setting', 
    #          transform=plt.gca().transAxes, ha='center', va='top', 
    #          fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{parent_model}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_feature_type_analysis(baseline_df, distillation_df, output_dir):
    """Create analysis plots for continuous-heavy vs categorical-heavy datasets."""
    
    # Define dataset categories
    continuous_heavy = ['diabetes', 'blood', 'calhousing', 'jungle']
    categorical_heavy = ['compas', 'covertype']
    
    # Merge all baseline and distillation data
    merged_all = merge_baseline_distillation(baseline_df, distillation_df)
    
    # Exclude CARTE results
    merged_all = merged_all[merged_all.get('parent_model', 'baseline') != 'carte'].copy()
    
    # Filter for each dataset type
    continuous_data = merged_all[merged_all['dataset'].isin(continuous_heavy)].copy()
    categorical_data = merged_all[merged_all['dataset'].isin(categorical_heavy)].copy()
    
    # Create figure with 2 subplots
    plt.figure(figsize=(16, 6))
    
    # Define shot values and student model order
    shot_values = [4, 8, 16, 32, 64, 128, 256]
    xtick_values = [4, 16, 64, 256]
    student_model_order = ['decision_tree', 'log. reg.', 'log. rule regr.', 'ttnet', 'xgboost']
    
    for idx, (data, title, dataset_type) in enumerate([
        (continuous_data, 'Continuous-Heavy Datasets\n(diabetes, blood, calhousing, jungle)', 'continuous'),
        (categorical_data, 'Categorical-Heavy Datasets\n(compas, covertype)', 'categorical')
    ]):
        plt.subplot(1, 2, idx + 1)
        
        if len(data) == 0:
            plt.text(0.5, 0.5, f'No data available for\n{dataset_type} datasets', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title(title, fontsize=16, fontweight='bold')
            continue
        
        # Group by student model, numshot, and method, then average across parent models and datasets
        grouped_data = data.groupby(['student_model', 'numshot', 'method'])['mean_auc'].mean().reset_index()
        
        # Filter to only include models that exist in the data
        student_models = [model for model in student_model_order if model in grouped_data['student_model'].unique()]
        colors = plt.cm.Set1(np.linspace(0, 1, len(student_models)))
        
        # Plot lines for each student model
        for i, student in enumerate(student_models):
            student_data = grouped_data[grouped_data['student_model'] == student]
            
            # Baseline data
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
                            linewidth=2, markersize=6)
            
            # Distillation data
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
                            linewidth=2, markersize=6)
        
        # Create legend (only for the first subplot to avoid duplication)
        if idx == 0:
            legend_elements = []
            # Add student model colors
            for i, student in enumerate(student_models):
                legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=3, label=student))
            # Add line style explanations
            legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Baseline'))
            legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Distillation (ours)'))
            plt.legend(handles=legend_elements, loc='best', fontsize=12)
        
        # Set ticks and formatting
        plt.gca().set_xticks(shot_values, minor=True)
        plt.gca().set_xticks(xtick_values, minor=False)
        plt.gca().tick_params(axis='x', which='minor', length=6, width=1)
        plt.gca().tick_params(axis='x', which='major', length=8, width=2)
        plt.xticks(xtick_values, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(min(shot_values) * 0.8, max(shot_values) * 1.1)
        plt.xlabel('Number of Shots', fontsize=14)
        plt.ylabel('Mean AUC', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Performance Analysis by Dataset Feature Type', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created feature type analysis plot: feature_type_analysis.png")


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
    
    # Create feature type analysis
    print("\nCreating feature type analysis...")
    create_feature_type_analysis(baseline_df, distillation_df, output_dir)



if __name__ == "__main__":
    main()
