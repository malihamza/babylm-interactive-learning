# Interactive LLM Pretraining & PPO Fine-Tuning Framework

This repository provides a modular framework for 
**language model pretraining** followed by 
**reinforcement learning-based fine-tuning** using 
**Proximal Policy Optimization (PPO)**. 
It supports structured dataset preparation, teacher-based reward modeling, and fine-tuning with `trl`'s `PPOTrainer`.

---

## 1. Setup & Installation

### Create Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # for Linux/Mac
# OR
.venv\Scripts\activate     # for Windows
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Pretraining

Before applying reinforcement learning, you can optionally **pretrain a language model** on a domain-specific dataset using standard language modeling objectives.

### Example Command

```bash
python3 src/pre_training/training.py \
  --hub_repo llm-slice/babylm-gpt2-small \
  --dataset_fraction 1 \
  --epochs 10 \
  --batch_size 32 \
  --grad_accum_steps 4 \
  --bf16
```

**Arguments:**

- `--hub_repo` : Name of the Hugging Face model repo to push to
- `--dataset_fraction` : Fraction of the dataset to use (1 = full)
- `--epochs` : Number of pretraining epochs
- `--batch_size` : Training batch size
- `--grad_accum_steps` : Gradient accumulation steps
- `--bf16` : Enable bfloat16 precision (if supported)

This pretraining step ensures that the model is well-initialized before applying PPO-based fine-tuning.

On a cluster, execute ```run_pretrain.sh```, which creates a job by calling ```run_pretrain.hpc```, which then calls ```src/pre_training/training.py```.

## 3. Teacher Model Configuration

Teacher model evaluates prompt-response pairs and assigns scores used as reward signals during PPO training. Configuration is stored in:

```
config/teacher.yaml
```

### Sample Fields:

```yaml
model_name_or_path: "meta-llama/Llama-3-8b-instruct"
prompt_template_path: "templates/rating_prompt.jinja"
temperature: 0.0
max_tokens: 32
min_score: 1
max_score: 9
```

Ensure your `prompt_template_path` is a valid markdown (.md) file, including the '{{student_completion}}' to evaluate.

---

## 4. Dataset Configuration

The dataset builder (`DeterministicPromptDatasetBuilder`, `TinyStoriesDatasetBuilder`, `WritingPromptsDatasetBuilder`, `IMDBDatasetBuilder`, etc.) handles dataset loading, filtering, and formatting. Token limit ensures sampling only up to a defined number of tokens for PPO training.

Each dataset builder is added to the `DatasetCombiner`, like:

```python
builder = TinyStoriesDatasetBuilder(...)
combined_dataset = DatasetCombiner([builder])
combined_dataset.set_token_limit(token_limit=1000000)
combined_dataset = combined_dataset.load()
```

- Token limit ensures consistent training size
- Each data sample has: `input_ids`, `query`, and `num_tokens`

---

## 5. PPO Training Configuration

All PPO training settings are defined in:

```
config/ppo.yaml
```

### Sample Fields:

```yaml
model_name: llm-slice/blm-gpt2s-90M-s42
revision_name: chck_900M
learning_rate: 1e-6
log_with: wandb
hf_org : llm-slice

# Trainer Settings
num_epochs: 1
batch_size: 20
token_limit: 20000
save_base_dir: saved_models/

data_path: data/ppo/

# Output generation
generation_kwargs:
  min_length: -1
  max_new_tokens: 90
  top_k: 0
  top_p: 1
  do_sample: true
  num_beams: 1

```

These fields are passed to `CustomPPOConfig`, which extends `trl.PPOConfig` and adds extra custom logic such as checkpointing and token counting.

---

## 6. Run PPO Fine-Tuning

```bash
python ppo.py
```

This will:
- Load datasets
- Load reward model (either random or teacher-based)
- Fine-tune using PPO with checkpointing
- Log training metadata and rewards in the `saved_models/` folder

---

## 7. Output Folder Structure

During PPO training, the code automatically creates a dedicated output directory to store model checkpoints and training metadata.

### Folder Naming Convention

The folder name follows this format:

```
<model_name>_<X_tokens>__<YYYY-MM-DD__HH-MM-SS>/
```

**Example:**
```
blm-gpt2s-90M-s42_chck_20M_ppo-1000K-seed42__2025-08-14__12-11-06/
```

This ensures each training run has a unique and timestamped folder.

### Folder Structure

Inside this folder:

```
blm-gpt2s-90M-s42_chck_20M_ppo-1000K-seed42__2025-08-14__12-11-06/
│
└── meta_data/
    ├── ppo.yaml                 # PPO config file
    ├── teacher.yaml             # Teacher config file
    ├── generated_outputs.csv    # Prompt → Response + reward log
    └── training_stats.csv       # Batch-wise reward and loss tracking
```


## 8. Evaluation

After pretraining or interactive reinforcement learning of your model, you can evaluate it using the [BaybLM evaluation-pipeline-2025](https://github.com/babylm/evaluation-pipeline-2025) scripts.

### Step-by-Step Instructions

- `eval_finetuning.sh`: Evaluates your **fine-tuned model** on the benchmark tasks  
- `eval_zero_shot.sh`: Evaluates your model in a **zero-shot setting** (no examples are shown during evaluation)
- `eval_zero_shot_fast.sh`: Evaluates your model on a number of checkpoints in a **zero-shot setting** (no examples are shown during evaluation), on reduced dataset
- `eval_aoa.sh`: Evaluates your model on all checkpoints for age of acquisition (AoA).

On a cluster, you can use the `run_(...).sh` to run a job with `run_(...).hpc`.