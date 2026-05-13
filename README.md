# multilingual-bridge-heads

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
Step 1  build_multilingual          Translate prompts into 8 languages via Wikidata + templates
        │                           → data/processed/{lang}/two_hop_{lang}.json
        ▼
Step 2  evaluate_multilingual       Run LLM inference, record per-sample correctness
        │                           → data/{model}/eval/
        ▼
Step 2.5 activation_patching        Layer-wise cross-lingual activation patching (Pearson correlation)
        │                           → output/step2_5_activation_patching/
        ▼
Step 3  filter_multilingual         Keep samples that are 2-hop correct but not shortcut-solvable
        │                           → data/{model}/filtered/
        ▼
Step 4  Bridge_head_Score (BHS)     Gradient-based head importance: z(FH) + z(TH) − z(SH)
        │                           Identify General (intersection) and Specific (residual top-%) heads
        │                           → output/step4_filtering_Bridge_head_Score/
        ▼
Step 5  Ablation + Patchscopes      Mean ablation filtering + Patchscopes residual stream verification
        │                           → output/step5_BridgeHead_Ablation_Patchscopes/
        │                           → final_bridge_heads.json  (per model)
        ▼
Step 6  Validation                  Causal validation via 4 stages:
                                      (a) Jaccard overlap analysis
                                      (b) Mean ablation NLL test
                                      (c) Scaling amplification accuracy
                                      (d) Cross-lingual transfer accuracy
                                    → output/raw_data/3_Head_Validation/
```

---

## Repository Structure

```
multilingual_hop/
├── script/
│   ├── step1_build_multilingual.py
│   ├── step2_evaluate_multilingual.py
│   ├── step2_5_activation_patching.py
│   ├── step3_filter_multilingual.py
│   ├── step4_filtering_Bridge_head_Score.py
│   ├── step5_BridgeHead_Ablation_Patchscopes.py
│   ├── step6_Bridge_Head_validation.py
│   ├── visualization.py
│   ├── run_experience.py          # batch runner for chaining commands
│   ├── config/
│   │   └── relation_templates.json
│   └── utils/
├── data/
│   └── raw/two_hop.csv            # source dataset (HoppingTooLate)
│
└── .gitignore
```

---

## Environment Setup

Python **3.12** is required. The code is tested on a multi-GPU server with CUDA.

```bash
# 1. Create and activate a virtual environment
python3 -m venv env_multilingual_hop
source env_multilingual_hop/bin/activate

# 2. Install dependencies
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==5.5.4 accelerate
pip install pandas scipy scikit-learn tqdm
```

HuggingFace model access (LLaMA 3.1 requires a gated token):

```bash
huggingface-cli login   # paste your HF token when prompted
```

---

## Running the Pipeline

Each step is run independently. Pass `--model` (HuggingFace model ID) and `--model-short` (short name used for output directories).

```bash
# Step 1 — Build multilingual dataset (CPU, one-time)
python script/step1_build_multilingual.py

# Step 2 — LLM evaluation (GPU)
python script/step2_evaluate_multilingual.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70

# Step 3 — Filter valid samples (CPU)
python script/step3_filter_multilingual.py --model-short llama31_70

# Step 4 — Bridge Head Score (GPU, run FH / TH / SH conditions separately, then aggregate)
python script/step4_filtering_Bridge_head_Score.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70 \
    --langs en ko zh ja es --condition FH

python script/step4_filtering_Bridge_head_Score.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70 \
    --langs en ko zh ja es --condition aggregate-only

# Step 5 — Ablation filtering + Patchscopes (GPU)
python script/step5_BridgeHead_Ablation_Patchscopes.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70 \
    --langs ko zh ja es --stage all

# Step 6 — Validation (GPU; Jaccard stage is CPU-only)
python script/step6_Bridge_Head_validation.py \
    --model meta-llama/Llama-3.1-70B --model-short llama31_70 \
    --langs ko zh ja es
```

For Qwen2.5-72B, replace `--model meta-llama/Llama-3.1-70B --model-short llama31_70` with `--model Qwen/Qwen2.5-72B --model-short qwen25_72` throughout.

To chain multiple commands sequentially, edit and run `script/run_experience.py`.

---

## Data

The base dataset is [HoppingTooLate](https://github.com/Maboroshi-ai/HoppingTooLate) (`data/raw/two_hop.csv`).  
Processed multilingual JSONs (~650 MB) are not tracked in this repository. Run Step 1 to regenerate them.

---

## Results Summary

Final validation results are in `output/raw_data/3_Head_Validation/`.  
See [Bridge_Head_Validation_Report.md](Bridge_Head_Validation_Report.md) for a full analysis report.
