from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Paste the commands you run in the terminal here as-is.
# Example:
# COMMANDS = [
#     "python script/step2_evaluate_multilingual.py --model ... --model-short ...",
#     "python script/step3_filter_multilingual.py --model-short ...",
# ]
COMMANDS = [
    # Step 1: Build multilingual dataset
    "python script/step1_build_multilingual.py --langs en ko zh ja es",

    # Step 2: Evaluate multilingual (LLaMA → Qwen)
    "python script/step2_evaluate_multilingual.py --model meta-llama/Llama-3.1-70B --model-short llama31_70 --langs en ko zh ja es",
    "python script/step2_evaluate_multilingual.py --model Qwen/Qwen2.5-72B --model-short qwen25_72 --langs en ko zh ja es",

    # Step 2.5: Activation patching (attn + mlp, LLaMA → Qwen)
    "python script/step2_5_activation_patching.py --model meta-llama/Llama-3.1-70B --model-short llama31_70 --patch-type attn --langs en ko zh ja es",
    "python script/step2_5_activation_patching.py --model meta-llama/Llama-3.1-70B --model-short llama31_70 --patch-type mlp --langs en ko zh ja es",
    "python script/step2_5_activation_patching.py --model Qwen/Qwen2.5-72B --model-short qwen25_72 --patch-type attn --langs en ko zh ja es",
    "python script/step2_5_activation_patching.py --model Qwen/Qwen2.5-72B --model-short qwen25_72 --patch-type mlp --langs en ko zh ja es",

    # Extra (Pearson correlation of patching profiles, runs after Step 2.5)
    "python script/extra/step2_5_pearson_correlation.py --models llama31_70 qwen25_72 --langs en ko zh ja es",

    # Step 3: Filter multilingual
    "python script/step3_filter_multilingual.py --model-short llama31_70 --langs en ko zh ja es",
    "python script/step3_filter_multilingual.py --model-short qwen25_72 --langs en ko zh ja es",

    # Step 4: Bridge Routing Score filtering
    "python script/step4_filtering_Bridge_Routing_Score.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es",
    "python script/step4_filtering_Bridge_Routing_Score.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es",

    # Step 5: Bridge Head Ablation + Patchscopes
    "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage all",
    "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage all",

    # Step 6: Bridge Head validation
    "python script/step6_Bridge_Head_validation.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage all",
    "python script/step6_Bridge_Head_validation.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage all",

    # Extra 1: Suppressor ablation
    "python script/extra/extra_1_suppressor_ablation.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es",
    "python script/extra/extra_1_suppressor_ablation.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es",

    # Extra 2: Stronger filtering
    "python script/extra/extra_2_stronger_filtering.py --models llama31_70 qwen25_72 --langs ko zh ja es --metric successes --min-successes 4",

    # Extra 3: BRS variants (variants → compare → ablation per model)
    "python script/extra/extra_3_brs_variants.py --stage variants --models llama31_70 qwen25_72 --langs ko zh ja es",
    "python script/extra/extra_3_brs_variants.py --stage compare --models llama31_70 qwen25_72 --langs ko zh ja es",
    "python script/extra/extra_3_brs_variants.py --stage ablation --models llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es",
    "python script/extra/extra_3_brs_variants.py --stage ablation --models qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es",
]


def run_command(command: str) -> None:
    print(f"\n$ {command}\n", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    for command in COMMANDS:
        run_command(command)


if __name__ == "__main__":
    main()
