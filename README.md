# xhop-bridge-heads

Identifying **language-general and language-specific Bridge Heads** — attention heads that mediate the retrieval of intermediate bridge entities in cross-lingual multi-hop reasoning.

Experiments are conducted on **LLaMA 3.1 70B** and **Qwen2.5 72B** across four non-English languages (Korean, Chinese, Japanese, Spanish).

---

## What is a Bridge Head?

In a 2-hop question (e.g. *"What is the capital of the country where [Person] was born?"*), the model must first retrieve an intermediate entity (e2, the country) before answering.  
A **Bridge Head** is an attention head that plays a causal role in this intermediate retrieval step.

We decompose Bridge Heads into two types:

| Type | Description |
|---|---|
| **General** | Language-agnostic; active across all languages |
| **Specific** | Language-dedicated; active only for a particular language |

---

## Pipeline

```
data/raw/two_hop.csv  (HoppingTooLate dataset)
        │
        ▼
Step 1  build_multilingual          Translate prompts into target languages via Wikidata + templates
        │                           → data/processed/{lang}/two_hop_{lang}.json
        ▼
Step 2  evaluate_multilingual       Run LLM inference, record per-sample correctness
        │                           → data/{model}/eval/
        ▼
Step 2.5 activation_patching        Layer-wise cross-lingual activation patching (attn + mlp)
        │                           → output/step2_5_activation_patching/
        │
        ├── [Extra] pearson_correlation   Pearson correlation of cross-lingual patching profiles
        │                                → output/extra/step2_5_pearson_correlation/
        ▼
Step 3  filter_multilingual         Keep samples that are 2-hop correct but not shortcut-solvable
        │                           → data/{model}/filtered/
        ▼
Step 4  Bridge_Routing_Score (BRS)   Gradient-based head importance: z(FH) + z(TH) − z(SH)
        │                           Identify General (intersection) and Specific (residual top-%) heads
        │                           → output/step4_filtering_Bridge_Routing_Score/
        ▼
Step 5  Ablation + Patchscopes      Mean ablation filtering + Patchscopes residual stream verification
        │                           → output/step5_BridgeHead_Ablation_Patchscopes/
        │                           → final_bridge_heads.json  (per model)
        ▼
Step 6  Validation                  Causal validation via 4 stages:
        │                             (a) Jaccard overlap analysis
        │                             (b) Mean ablation NLL test
        │                             (c) Scaling amplification accuracy
        │                             (d) Cross-lingual transfer accuracy
        │                           → output/step6_Bridge_Head_validation/
        │
        ├── [Extra 1] suppressor_ablation    Ablation of suppressor heads
        ├── [Extra 2] stronger_filtering     Stricter patchscopes-based head filtering
        └── [Extra 3] brs_variants           Alternative BRS formula variants + Jaccard analysis
```

---

## Repository Structure

```
xhop-bridge-heads/
├── script/
│   ├── step1_build_multilingual.py
│   ├── step2_evaluate_multilingual.py
│   ├── step2_5_activation_patching.py
│   ├── step3_filter_multilingual.py
│   ├── step4_filtering_Bridge_Routing_Score.py
│   ├── step5_BridgeHead_Ablation_Patchscopes.py
│   ├── step6_Bridge_Head_validation.py
│   ├── run_experience.py              # Batch runner — edit COMMANDS and run to execute the full pipeline
│   ├── config/
│   │   └── relation_templates.json
│   ├── extra/
│   │   ├── step2_5_pearson_correlation.py
│   │   ├── extra_1_suppressor_ablation.py
│   │   ├── extra_2_stronger_filtering.py
│   │   └── extra_3_brs_variants.py
│   └── utils/
│       ├── bridge_utils.py
│       ├── common.py
│       ├── head_hooks.py
│       ├── model_utils.py
│       ├── prompt_utils.py
│       └── wikidata_utils.py
├── data/
│   └── raw/two_hop.csv                # Source dataset (HoppingTooLate)
└── .gitignore
```

---

## Environment Setup

Python **3.10+** is required. The code is tested on a multi-GPU server with CUDA.

```bash
# 1. Create and activate a virtual environment
python3 -m venv env_xhop
source env_xhop/bin/activate          # Windows: env_xhop\Scripts\activate

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate
pip install pandas scipy scikit-learn tqdm
```

HuggingFace model access (LLaMA 3.1 requires a gated token):

```bash
huggingface-cli login   # paste your HF token when prompted
```

---

## Running the Pipeline

The easiest way to run the full pipeline is via `script/run_experience.py`.  
The `COMMANDS` list is pre-populated with every step in the correct order. Simply run:

```bash
python script/run_experience.py
```

Each command in `COMMANDS` is executed sequentially. To run a subset, comment out the steps you do not need.

### What `run_experience.py` runs (in order)

| # | Script | Description |
|---|--------|-------------|
| 1 | `step1_build_multilingual.py` | Build multilingual dataset |
| 2–3 | `step2_evaluate_multilingual.py` | LLM evaluation (LLaMA → Qwen) |
| 4–7 | `step2_5_activation_patching.py` | Activation patching (attn + mlp × 2 models) |
| 8 | `extra/step2_5_pearson_correlation.py` | Pearson correlation of patching profiles |
| 9–10 | `step3_filter_multilingual.py` | Filter valid 2-hop samples |
| 11–12 | `step4_filtering_Bridge_Routing_Score.py` | Bridge Routing Score computation |
| 13–14 | `step5_BridgeHead_Ablation_Patchscopes.py` | Ablation + Patchscopes verification |
| 15–16 | `step6_Bridge_Head_validation.py` | Causal validation (all stages) |
| 17–18 | `extra/extra_1_suppressor_ablation.py` | Suppressor head ablation |
| 19 | `extra/extra_2_stronger_filtering.py` | Stronger patchscopes filtering |
| 20–23 | `extra/extra_3_brs_variants.py` | BRS formula variants (variants → compare → ablation) |

### Running steps individually

If you prefer to run steps one at a time:

```bash
# Step 1 — Build multilingual dataset (CPU, one-time)
python script/step1_build_multilingual.py --langs en ko zh ja es

# Step 2 — LLM evaluation (GPU)
python script/step2_evaluate_multilingual.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70 --langs en ko zh ja es
python script/step2_evaluate_multilingual.py \
    --model Qwen/Qwen2.5-72B --model-short qwen25_72 --langs en ko zh ja es

# Step 2.5 — Activation patching (GPU)
python script/step2_5_activation_patching.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70 --patch-type attn --langs en ko zh ja es
python script/step2_5_activation_patching.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70 --patch-type mlp --langs en ko zh ja es
# (repeat with --model Qwen/Qwen2.5-72B --model-short qwen25_72)

# Step 3 — Filter valid samples (CPU)
python script/step3_filter_multilingual.py --model-short llama31_70 --langs en ko zh ja es

# Step 4 — Bridge Routing Score (GPU)
python script/step4_filtering_Bridge_Routing_Score.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70 --langs ko zh ja es

# Step 5 — Ablation + Patchscopes (GPU)
python script/step5_BridgeHead_Ablation_Patchscopes.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70 \
    --langs ko zh ja es --stage all

# Step 6 — Validation (GPU; Jaccard stage is CPU-only)
python script/step6_Bridge_Head_validation.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70 \
    --langs ko zh ja es --stage all
```

---

## Data

The base dataset is [HoppingTooLate](https://github.com/Maboroshi-ai/HoppingTooLate) (`data/raw/two_hop.csv`).  
Processed multilingual JSONs (~650 MB) are not tracked in this repository. Run Step 1 to regenerate them.

---

## Results Summary

Final validation results are saved to `output/step6_Bridge_Head_validation/`.
