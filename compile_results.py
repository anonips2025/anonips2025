import os
import pandas as pd
import re
from collections import defaultdict


def compile_distillation_results(base_dir="eval_res", output_csv="distillation_results.csv"):
    """
    Compile all distillation results from eval_res/{parent_model}/{dataset_name}/{numshot}_shot/{student_model}/results.txt
    into a single CSV file, extracting mean AUC and mean complexity.
    """
    all_results = []
    
    # Get all parent models (excluding baselines)
    parent_models = [d for d in os.listdir(base_dir) if d != 'baselines' and os.path.isdir(os.path.join(base_dir, d))]
    
    for parent_model in parent_models:
        parent_path = os.path.join(base_dir, parent_model)
        
        # Get all datasets
        datasets = [d for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]
        
        for dataset in datasets:
            dataset_path = os.path.join(parent_path, dataset)
            
            # Get all shot configurations
            shot_configs = [d for d in os.listdir(dataset_path) if d.endswith('_shot') and os.path.isdir(os.path.join(dataset_path, d))]
            
            for shot_config in shot_configs:
                numshot = shot_config.replace('_shot', '')
                shot_path = os.path.join(dataset_path, shot_config)
                
                # Get all student models
                student_models = [d for d in os.listdir(shot_path) if os.path.isdir(os.path.join(shot_path, d))]
                
                for student_model in student_models:
                    results_file = os.path.join(shot_path, student_model, 'results.txt')
                    
                    if os.path.exists(results_file):
                        try:
                            with open(results_file, 'r') as f:
                                content = f.read()
                            
                            # Extract mean AUC and mean complexity
                            mean_auc = None
                            mean_complexity = None
                            
                            for line in content.split('\n'):
                                if line.startswith('Mean AUC:'):
                                    mean_auc = float(line.split(':')[1].strip())
                                elif line.startswith('Mean Complexity:'):
                                    complexity_str = line.split(':')[1].strip()
                                    if complexity_str != 'N/A':
                                        mean_complexity = float(complexity_str)
                            
                            # Store the result
                            result = {
                                'parent_model': parent_model,
                                'dataset': dataset,
                                'numshot': int(numshot),
                                'student_model': student_model,
                                'mean_auc': mean_auc,
                                'mean_complexity': mean_complexity
                            }
                            all_results.append(result)
                            
                        except Exception as e:
                            print(f"Error processing {results_file}: {e}")
    
    # Convert to DataFrame and save
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"Compiled {len(all_results)} results into {output_csv}")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        return df
    else:
        print("No distillation results found")
        return None


def merge_methods(dir_path, input_merged_csv="merged_baseline_summary.csv", output_merged_csv="merged_all.csv"):

    all_dfs = []
    all_methods = os.listdir(dir_path)
    for method in all_methods:
        csv_path = os.path.join(dir_path, method, input_merged_csv)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['method'] = method
            all_dfs.append(df)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(os.path.join(dir_path, output_merged_csv), index=False)
        print(f"Merged CSV saved to {output_merged_csv}")
    else:
        print("No experiment CSVs found to merge.")


def compile_baseline_results(base_dir="eval_res/baselines", output_csv="baseline_results.csv"):
    """
    Compile all baseline results from eval_res/baselines/{dataset_name}/{numshot}_shot/{model_name}/results.txt
    into a single CSV file, extracting mean AUC and mean complexity.
    """
    all_results = []
    
    if not os.path.exists(base_dir):
        print(f"Baseline directory {base_dir} does not exist")
        return None
    
    # Get all datasets
    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        
        # Get all shot configurations
        shot_configs = [d for d in os.listdir(dataset_path) if d.endswith('_shot') and os.path.isdir(os.path.join(dataset_path, d))]
        
        for shot_config in shot_configs:
            numshot = shot_config.replace('_shot', '')
            shot_path = os.path.join(dataset_path, shot_config)
            
            # Get all baseline models
            baseline_models = [d for d in os.listdir(shot_path) if os.path.isdir(os.path.join(shot_path, d))]
            
            for baseline_model in baseline_models:
                results_file = os.path.join(shot_path, baseline_model, 'results.txt')
                
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'r') as f:
                            content = f.read()
                        
                        # Extract mean AUC and mean complexity
                        mean_auc = None
                        mean_complexity = None
                        
                        for line in content.split('\n'):
                            if line.startswith('Mean AUC:'):
                                mean_auc = float(line.split(':')[1].strip())
                            elif line.startswith('Mean Complexity:'):
                                complexity_str = line.split(':')[1].strip()
                                if complexity_str != 'N/A':
                                    mean_complexity = float(complexity_str)
                        
                        # Store the result
                        result = {
                            'dataset': dataset,
                            'numshot': int(numshot),
                            'baseline_model': baseline_model,
                            'mean_auc': mean_auc,
                            'mean_complexity': mean_complexity
                        }
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"Error processing {results_file}: {e}")
    
    # Convert to DataFrame and save
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"Compiled {len(all_results)} baseline results into {output_csv}")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        return df
    else:
        print("No baseline results found")
        return None


if __name__ == "__main__":
    # Compile all distillation results into a single CSV
    print("Compiling distillation results...")
    df = compile_distillation_results(base_dir='eval_res', output_csv='distillation_results.csv')
    
    if df is not None:
        print("\nSample of compiled results:")
        print(df.head())
        print(f"\nUnique parent models: {df['parent_model'].unique()}")
        print(f"Unique datasets: {df['dataset'].unique()}")
        print(f"Unique shot counts: {sorted(df['numshot'].unique())}")
        print(f"Unique student models: {df['student_model'].unique()}")
    
    # Compile baseline results
    print("\n" + "="*50)
    print("Compiling baseline results...")
    baseline_df = compile_baseline_results(base_dir='eval_res/baselines', output_csv='baseline_results.csv')
    
    if baseline_df is not None:
        print("\nSample of baseline results:")
        print(baseline_df.head())
        print(f"\nUnique datasets: {baseline_df['dataset'].unique()}")
        print(f"Unique shot counts: {sorted(baseline_df['numshot'].unique())}")
        print(f"Unique baseline models: {baseline_df['baseline_model'].unique()}")