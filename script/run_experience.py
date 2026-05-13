from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 여기에 터미널에서 치던 명령어를 문자열 그대로 넣으면 됩니다.
# 예:
# COMMANDS = [
#     "python script/step2_evaluate_multilingual.py --model ... --model-short ...",
#     "python script/step3_filter_multilingual.py --model-short ...",
# ]
COMMANDS = [
    # "python script/step2_evaluate_multilingual.py --model meta-llama/Llama-3.1-70B --model-short llama31_70",
    # "python script/step2_evaluate_multilingual.py --model Qwen/Qwen2.5-72B --model-short qwen25_72",
    "python script/step2_5_activation_patching.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --patch-type attn",
    "python script/step2_5_activation_patching.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --patch-type mlp",
    "python script/step2_5_activation_patching.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --patch-type attn",
    "python script/step2_5_activation_patching.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --patch-type mlp",
    # "python script/step3_filter_multilingual.py --model-short llama31_70",
    # "python script/step3_filter_multilingual.py --model-short qwen25_72",
    # "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage attention",
    # "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage all",
    # "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage all",
    # "python script/step5_BridgeHead_Ablation_Patchscopes.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage all",
    # "python script/step6_Bridge_Head_validation.py --model-short llama31_70 --model meta-llama/Llama-3.1-70B --langs ko zh ja es --stage ablation",
    # "python script/step6_Bridge_Head_validation.py --model-short qwen25_72 --model Qwen/Qwen2.5-72B --langs ko zh ja es --stage ablation",
]


def run_command(command: str) -> None:
    print(f"\n$ {command}\n", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    for command in COMMANDS:
        run_command(command)


if __name__ == "__main__":
    main()
