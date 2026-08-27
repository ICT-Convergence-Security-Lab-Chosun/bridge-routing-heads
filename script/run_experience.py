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
    # # Step 1: Build multilingual dataset
    # "python script/step1_build_multilingual.py --langs en ko zh ja es",

    # # Step 2: Evaluate multilingual (LLaMA → Qwen)
    # "python script/step2_evaluate_multilingual.py --model meta-llama/Llama-3.1-70B --model-short llama31_70 --langs en ko zh ja es",
    # "python script/step2_evaluate_multilingual.py --model Qwen/Qwen2.5-72B --model-short qwen25_72 --langs en ko zh ja es",

    # # Step 2.5: Activation patching (attn + mlp, LLaMA → Qwen)
    # "python script/step2_5_activation_patching.py --model meta-llama/Llama-3.1-70B --model-short llama31_70 --patch-type attn --langs en ko zh ja es",
    # "python script/step2_5_activation_patching.py --model meta-llama/Llama-3.1-70B --model-short llama31_70 --patch-type mlp --langs en ko zh ja es",
    # "python script/step2_5_activation_patching.py --model Qwen/Qwen2.5-72B --model-short qwen25_72 --patch-type attn --langs en ko zh ja es",
    # "python script/step2_5_activation_patching.py --model Qwen/Qwen2.5-72B --model-short qwen25_72 --patch-type mlp --langs en ko zh ja es",

    # # Extra (Pearson correlation of patching profiles, runs after Step 2.5)
    # "python script/extra/step2_5_pearson_correlation.py --models llama31_70 qwen25_72 --langs en ko zh ja es",

    # # Step 3: Filter multilingual
    # "python script/step3_filter_multilingual.py --model-short llama31_70 --langs en ko zh ja es",
    # "python script/step3_filter_multilingual.py --model-short qwen25_72 --langs en ko zh ja es",

    # # Step 4: Bridge Routing Score filtering
    # "python script/step4_filtering_Bridge_Routing_Score.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es",
    # "python script/step4_filtering_Bridge_Routing_Score.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es",

    # # Step 5: Bridge Head Ablation + Patchscopes
    # "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage all",
    # "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage all",

    # # Step 6: Bridge Head validation
    # "python script/step6_Bridge_Head_validation.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --head-selection random --stage scaling --alpha-list 1.0",
    # "python script/step6_Bridge_Head_validation.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --head-selection random --stage transfer --alpha-list 1.0",
    # "python script/step6_Bridge_Head_validation.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --head-selection random --stage scaling --alpha-list 1.0",
    # "python script/step6_Bridge_Head_validation.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --head-selection random --stage transfer --alpha-list 1.0",

    # # Extra 1: Suppressor ablation    
    # "python script/extra/extra_1_suppressor_ablation.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es",
    # "python script/extra/extra_1_suppressor_ablation.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es",

    # # Extra 2: Stronger filtering
    # "python script/extra/extra_2_stronger_filtering.py --models llama31_70 qwen25_72 --langs ko zh ja es --metric successes --min-successes 4",

    # # Extra 3: BRS variants (variants → compare → ablation per model)
    # "python script/extra/extra_3_brs_variants.py --stage variants --models llama31_70 qwen25_72 --langs ko zh ja es",
    # "python script/extra/extra_3_brs_variants.py --stage compare --models llama31_70 qwen25_72 --langs ko zh ja es",
    # "python script/extra/extra_3_brs_variants.py --stage ablation --models llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es",
    # "python script/extra/extra_3_brs_variants.py --stage ablation --models qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es",

    # Extra 4: amplification effect on MMLU + already-correct cases (LLaMA & Qwen)
    # "python script/extra/extra_4_amplification_general_perf.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage mmlu_base",
    # "python script/extra/extra_4_amplification_general_perf.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage mmlu_amp",
    # "python script/extra/extra_4_amplification_general_perf.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage correct_specific",
    # "python script/extra/extra_4_amplification_general_perf.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage correct_general",

    # "python script/extra/extra_4_amplification_general_perf.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage mmlu_base",
    # "python script/extra/extra_4_amplification_general_perf.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage mmlu_amp",
    # "python script/extra/extra_4_amplification_general_perf.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage correct_specific",
    # "python script/extra/extra_4_amplification_general_perf.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage correct_general",

    # Extra 5: BRH activation-norm at the bridge token, success vs failure (run AFTER extra4)
    # "python script/extra/extra_5_bridge_activation_norm.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage all",
    # "python script/extra/extra_5_bridge_activation_norm.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage all",
    # "python script/step6_Bridge_Head_validation.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --head-selection random --stage scaling --alpha-list 1.0",
    # "python script/step6_Bridge_Head_validation.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --head-selection random --stage transfer --alpha-list 1.0",

    # Extra 6: localized-entity robustness of general BRH (rebuild data with target-language
    # entity anchors, re-run pipeline UP TO Stage-2 mean-ablation only, then compare vs original).
    # SMALL-SCALE: only records that were CORRECT in the original English-anchor run are
    # re-evaluated (failures WITH the anchor won't be fixed by localizing it, and head
    # identification only consumes correct samples anyway). ~32x less compute (~13k vs 410k).
    # (0) Full localized data was already built once (shared by both models):
    # "python script/step1_build_multilingual.py --langs en ko zh ja es --entity-prompt-label localized --output-dir data/processed_localized",
    # (1) Cut the per-model originally-correct subset (CPU, seconds):
    # "python script/extra/extra_6_prepare_localized_subset.py --model-short llama31_70 --langs en ko zh ja es",
    # # (2) LLaMA localized pipeline (namespace llama31_70_loc), up to Stage-2 ablation:
    # "python script/step2_evaluate_multilingual.py --model meta-llama/Llama-3.1-70B --model-short llama31_70_loc --langs en ko zh ja es --data-dir data/processed_localized_llama31_70 --batch-size 128 --resume",
    # "python script/step3_filter_multilingual.py --model-short llama31_70_loc --langs en ko zh ja es --data-dir data/processed_localized_llama31_70",
    # "python script/step4_filtering_Bridge_Routing_Score.py --model-short llama31_70_loc --model meta-llama/Llama-3.1-70B --langs ko zh ja es",
    # "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short llama31_70_loc --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage calibrate",
    # "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short llama31_70_loc --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage ablation",
    # # (4) Compare original vs localized Stage-2 BRH (add qwen25_72 to --models once its loc run is done):
    # "python script/extra/extra_6_localized_entity_bridge.py --models llama31_70 --langs ko zh ja es --localized-processed-dir data/processed_localized_llama31_70 --output-root output/extra/extra6_localized_bridge_llama31_70",
    # # (3) Qwen localized pipeline (run after LLaMA is verified):
    # "python script/extra/extra_6_prepare_localized_subset.py --model-short qwen25_72 --langs en ko zh ja es",
    "python script/step2_evaluate_multilingual.py --model Qwen/Qwen2.5-72B --model-short qwen25_72_loc --langs en ko zh ja es --data-dir data/processed_localized_qwen25_72 --batch-size 128 --resume",
    "python script/step3_filter_multilingual.py --model-short qwen25_72_loc --langs en ko zh ja es --data-dir data/processed_localized_qwen25_72",
    "python script/step4_filtering_Bridge_Routing_Score.py --model-short qwen25_72_loc --model Qwen/Qwen2.5-72B --langs ko zh ja es",
    "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short qwen25_72_loc --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage calibrate",
    # "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short qwen25_72_loc --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage ablation",
    # extra6 takes ORIGINAL model_short names (it appends _loc internally); one call per model
    # so the coverage columns read the matching per-model subset dir:
    # "python script/extra/extra_6_localized_entity_bridge.py --models qwen25_72 --langs ko zh ja es --localized-processed-dir data/processed_localized_qwen25_72 --output-root output/extra/extra6_localized_bridge_qwen25_72",
]


def run_command(command: str) -> None:
    print(f"\n$ {command}\n", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    for command in COMMANDS:
        run_command(command)


if __name__ == "__main__":
    main()
