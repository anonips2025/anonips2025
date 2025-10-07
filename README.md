# Towards Globally Interpretable Few-Shot Tabular Learning: Distilling Foundation Models

This repository contains the implementation for the paper **Towards Globally Interpretable Few-Shot Tabular Learning: Distilling Foundation Models** submitted to the NeurIPS 2025 Reliable ML for Unreliable Data Workshop.

## Environment Setup

**Install required packages:**
```bash
pip install -r env.txt
```


## Complete workflow

This section provides a step-by-step guide to reproduce all results. All the distillation and baseline results can be found at `eval_res/`. 

### Step 1: Getting the Datasets

```bash
python get_data.py
```

This script will:
- Download the datasets from OpenML using their dataset IDs
- Save each dataset as a CSV file in `dataset/{dataset_name}/{dataset_name}.csv`
- Generate metadata files (`{dataset_name}.info`) containing feature type information
- Skip datasets that already exist locally

The datasets will be organized in the following structure:
```
dataset/
├── albert/
│   ├── albert.csv
│   └── albert.info
└── ... (other datasets)
```

### Step 2: Generate Synthetic Data

To generate synthetic data for all datasets:

```bash
python src/synthetic_data_generator.py
```

**Output Structure:**
After running the script, each dataset folder will contain both original and synthetic data:
```
dataset/
├── bank/
│   ├── bank.csv              # Original data
│   ├── bank_synthetic.csv    # Generated synthetic data
│   └── bank.info             # Metadata
```

### Step 3: Inference

#### 3.1 TabLLM

##### Prerequisites: Setup [TabLLM](https://github.com/clinicalml/TabLLM) and [t-few](https://github.com/r-three/t-few) Repositories

1. **Clone the required repositories:**
```bash
# Clone TabLLM repository
git clone https://github.com/clinicalml/TabLLM.git

# Clone t-few repository  
git clone https://github.com/r-three/t-few.git
```

2. **Setup the t-few environment:**
```bash
conda create -n tfew python==3.7
conda activate tfew
pip install fsspec==2021.05.0
pip install torch datasets==2.0.0 transformers==4.15.0 pytorch-lightning==1.5.8 torchmetrics==0.6.2 psutil==5.9.0 deepspeed==0.5.10 sentencepiece==0.1.96 promptsource scipy
pip install urllib3==1.26.6
pip install importlib-metadata==4.13.0
pip install scikit-learn
```


##### Prepare Files and Datasets

1. **Serialize synthetic datasets for TabLLM:**
```bash
# Return to this project directory
cd /path/to/kdd25_test
python src/serialize_dataset.py
```

2. **Move necessary files to TabLLM and t-few directories:**
```bash
bash bin/tabllm_pre.sh
```

This script will:
- Copy dataset templates to the TabLLM templates directory
- Copy serialized datasets (both original and synthetic) to TabLLM datasets directory
- Copy modified t-few configuration files and scripts
- Set up the necessary file structure for TabLLM inference

##### Configure Paths

**Important:** Verify that paths are correctly set in your t-few installation:

1. **Check dataset path in `t-few/src/data/dataset_readers.py`:**
   - Line 75: Ensure `DATASETS_OFFLINE` points to your TabLLM datasets directory
   - Example: `DATASETS_OFFLINE = "/path/to/your/TabLLM/datasets_serialized"`

2. **Check template path in the same file:**
   - Line 233: Ensure the YAML template path points to your TabLLM templates directory
   - Example: `yaml_dict = yaml.load(open("/path/to/your/TabLLM/templates/templates_"))`

##### Run TabLLM Inference

Execute the TabLLM inference script:
```bash
# Navigate to t-few directory
cd /path/to/your/t-few

# Run few-shot inference on synthetic data
bash bin/few-shot-pretrained-synthetic-100k.sh
```


##### Extract and Organize Results

After inference is complete, extract the results:
```bash
# Return to this project directory
cd /path/to/kdd25_test

# Extract TabLLM inference results
bash bin/tabllm_post.sh
```

This script will:
- Copy inference results from t-few output to `eval_res/tabllm/`
- Run the extraction utility for each dataset and shot configuration
- Generate `y_pred.npy` and `X_synth.npy` files for downstream analysis

**Output Structure:**
```
eval_res/tabllm/
├── albert/
│   ├── 4_shot/
│   │   ├── t0.p           # Raw TabLLM results
│   │   ├── y_pred.npy     # Extracted predictions
│   │   └── X_synth.npy    # Synthetic features
│   └── 16_shot/
│       └── ... (similar structure)
└── ... (other datasets)
```


#### 3.2 TabM, TabPFN and CARTE Inference

After TabLLM inference, the next step is to run inference with TabM, TabPFN and CARTE models. These are simpler to execute as they don't require external repository setup.

```bash
bash bin/tabpfn.sh

bash bin/carte.sh

bash bin/tabm.sh
```

**Expected Output Structure:**
After running both scripts, your `eval_res/` directory will contain:
```
eval_res/
├── tabllm/
│   └── ... (from previous step)
├── tabm/
│   ├── albert/
│   │   ├── 4_shot/
│   │   │   └── commandline_args.txt
|   |   |   └── X_synth.npy
|   |   |   └── y_pred.npy
│   │   ├── 16_shot/
│   │   └── ... (other shot configs)
│   └── ... (other datasets)
├── tabpfn/
│   ├── albert/
│   │   ├── 4_shot/
│   │   │   └── commandline_args.txt
└── carte/
    ├── albert/
    │   ├── 4_shot/
    │   │   └── commandline_args.txt
    │   └── ... (other shot configs)
    └── ... (other datasets)
```


### Step 4: Distillation

After obtaining the parent models' results, distillation is performed with 5 student interpretable ML models: TTnet, XGBoost, Decision Tree, Logistic Regression, and Logistic Rule Regression (from the [aix360](https://aix360.readthedocs.io/en/latest/) toolkit).

```bash
python distillation_general.py
```

**Expected Output Structure:**
After running both scripts, your `eval_res/` directory will contain:
```
eval_res/
├── tabllm/
│   └── ...
├── tabpfn/
│   ├── albert/
│   │   ├── 4_shot/
│   |   │   ├── decision_tree/
│   |   │   |   └── results.txt
│   |   │   └── ... (other student models)
│   │   │   └── commandline_args.txt
|   |   |   └── X_synth.npy
|   |   |   └── y_pred.npy
│   │   ├── 16_shot/
│   │   └── ... (other shot configs)
│   └── ... (other datasets)
└── carte/
    ├── albert/
    │   ├── 8_shot/
    │   |   │   ├── decision_tree/
    |   │   |   └── results.txt
    |   │   └── ... (other student models)
    │   │   └── commandline_args.txt
    │   └── ... (other shot configs)
    └── ... (other datasets)
```


### Step 5: Generate baseline results

Next, we perform few-shot classification of the tabular datasets with the 5 interpretable ML models without distilling the parent models.

```bash
python baseline_general.py
```


### Step 6: Compile results and visualize

```bash
python compile_results.py

python visualize.py
```

Running `compile_results.py` will automatically generate `wilcoxon_test.txt` showing the Wilcoxon signed-rank statistical test mentioned in the paper with the obtained data.