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

Ensure your `prompt_template_path` is a valid Jinja2-style file using:
```
{{ context }} and {{ continuation }}
```

---

## 4. Dataset Configuration

The dataset builder (`TinyStoriesDatasetBuilder`, `IMDBDatasetBuilder`, etc.) handles dataset loading, filtering, and formatting. Token limit ensures sampling only up to a defined number of tokens for PPO training.

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
model_name: "meta-llama/Llama-3-8b-instruct"
learning_rate: 1.41e-5
batch_size: 2
log_with: null

# Extended Config Fields
token_limit: 1000000
checkpoint_interval: 100000
output_min_length: 64
output_max_length: 128
generation_kwargs:
  temperature: 1.0
  top_k: 50
  top_p: 0.95

num_epochs: 1
query_min_length: 32
query_max_length: 128
data_path: "data/"
save_base_dir: "saved_models"
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

During PPO training, the code automatically creates a dedicated output directory to store model checkpoints and training metadata. This is handled by the `make_model_output_dir()` function.

### Folder Naming Convention

The folder name follows this format:

```
<model_name>_<X_tokens>__<YYYY-MM-DD__HH-MM-SS>/
```

**Example:**
```
meta-llama_Llama-3-8b-instruct_300K_tokens__2025-07-20__23-15-10/
```

This ensures each training run has a unique and timestamped folder.

### Folder Structure

Inside this folder:

```
meta-llama_Llama-3-8b-instruct_300K_tokens__2025-07-20__23-15-10/
│
├── checkpoints/
│   ├── checkpoint_100K_tokens/
│   ├── checkpoint_200K_tokens/
│   └── ...
│
└── meta_data/
    ├── generated_outputs.csv    # Prompt → Response + reward log
    └── training_stats.csv       # Batch-wise reward and loss tracking
```

### Purpose of Each Subfolder:

- `checkpoints/`: Contains intermediate saved models at defined token intervals (e.g., every 100K tokens).
- `meta_data/`: Logs responses, rewards, and loss metrics for later analysis or reproducibility.

This design ensures easy tracking and analysis for each fine-tuning session, especially useful when training across different model configurations or datasets.

Happy fine-tuning!



## 8. Evaluation

After pretraining or fine-tuning your model, you can evaluate it using the [evaluation-pipeline-2025](https://github.com/babylm/evaluation-pipeline-2025) scripts.

### Step-by-Step Instructions

```bash
cd evaluation-pipeline-2025
./eval_finetuning.sh llm-slice/babylm-gpt2-small
./eval_zero_shot.sh llm-slice/babylm-gpt2-small
```

- `eval_finetuning.sh`: Evaluates your **fine-tuned model** on the benchmark tasks  
- `eval_zero_shot.sh`: Evaluates your model in a **zero-shot setting** (no examples are shown during evaluation)

Make sure your Hugging Face model (`llm-slice/babylm-gpt2-small`) is accessible or correctly logged in before running the scripts.

