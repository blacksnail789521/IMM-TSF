from tqdm import tqdm
import json
from pathlib import Path
from itertools import product
from tqdm import tqdm
from prettytable import PrettyTable
import time

from utils.tools import set_seed, print_formatted_dict
from main import get_args_from_parser, trainable

if __name__ == "__main__":
    """------------------------------------"""
    data_name_list = [
        # "GDELT",
        # "RepoHealth",
        # "MIMIC",
        # "FNSPID",
        # "ClusterTrace",
        # "StudentLife",
        # "ILINet",
        "CESNET",
        # "EPA-Air",
    ]

    model_name_list = [
        "Informer",
        "DLinear",
        "PatchTST",
        "TimesNet",
        "TimeMixer",
        "TimeLLM",
        "TTM",
        "CRU",
        "LatentODE",
        "NeuralFlow",
        "tPatchGNN",
    ]

    enable_text_list = [
        True,
        False,
    ]

    # use_text_embeddings = False
    use_text_embeddings = True

    TTF_module_list = [
        "TTF_RecAvg",
        "TTF_T2V_XAttn",
    ]

    MMF_module_list = [
        "MMF_GR_Add",
        "MMF_XAttn_Add",
    ]

    # llm_model_fusion = "GPT2"
    # llm_model_fusion = "BERT"
    # llm_model_fusion = "Llama"
    llm_model_fusion = "DeepSeek"

    llm_layers_fusion = None
    # llm_layers_fusion = 6

    split_method = "sample"
    # split_method = "instance"  # only for in-domain transfer learning

    tunable_params_path = None
    # tunable_params_path = Path(
    #     "exp_settings_and_results",
    #     "single_granularity",
    #     model_name,
    #     f"{data_name}.json",
    # )

    # batch_size = 1
    # batch_size = 2  # 8G
    batch_size = 8  # 24G
    # batch_size = 32
    # batch_size = 64
    # batch_size = 256
    """------------------------------------"""
    # Setup args
    args = get_args_from_parser()

    # Set all random seeds (Python, NumPy, PyTorch)
    set_seed(args.seed)

    # Generate combinations of data_name, model, and enable_text

    combinations = list(
        product(
            data_name_list,
            model_name_list,
            enable_text_list,
            TTF_module_list,
            MMF_module_list,
        )
    )

    run_times = []  # To store (description, time)
    total_start_time = time.time()

    for idx, combination in enumerate(
        tqdm(combinations, desc="Processing combinations")
    ):
        start_time_combination = time.time()  # Start timing for this combination

        if idx < 0:
            continue

        data_name, model_name, enable_text, TTF_module, MMF_module = combination
        print(
            f"\033[1;33mRunning combination {idx + 1}/{len(combinations)}: "
            f"data_name={data_name}, model_name={model_name}, "
            f"enable_text={enable_text}, TTF_module={TTF_module}, "
            f"MMF_module={MMF_module}, llm_model_fusion={llm_model_fusion}\033[0m"
        )

        # Skip non-default TTF/MMF modules when enable_text is False
        if not enable_text and (
            TTF_module != "TTF_RecAvg" or MMF_module != "MMF_GR_Add"
        ):
            print(
                f"\033[91mSkipping combination {idx + 1}: "
                f"TTF/MMF modules are not default when enable_text is False.\033[0m"
            )
            continue

        # Setup fixed params
        fixed_params = {
            "dataset": data_name,
            "model": model_name,
            "batch_size": batch_size,
            # "epoch": 1,
            "epoch": 1000,
            "enable_text": enable_text,
            "use_text_embeddings": use_text_embeddings,
            "split_method": split_method,
            "TTF_module": TTF_module,
            "MMF_module": MMF_module,
            "llm_model_fusion": llm_model_fusion,
            "llm_layers_fusion": llm_layers_fusion,
        }

        # Setup tunable params
        if tunable_params_path is None:
            tunable_params = {
                "lr": 1e-3,
                # "lr": 1e-4,
                "rec_ids": [
                    "10.0.178.41",
                    "router-1902",
                    "10.0.64.221",
                    "server-1245",
                    "firewall-3127",
                    "router-7669",
                    "10.0.212.206",
                    "10.0.31.13",
                    "10.0.90.172",
                    "switch-9995",
                    "firewall-8704",
                ],  # TODO: CESNET
            }

        # Create result directory if it doesn't exist
        results_dir = Path("experiment_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Build a unique filename for each combination
        result_filename = (
            f"{data_name}_{model_name}_enable_text={enable_text}"
            f"_TTF_module={TTF_module}_MMF_module={MMF_module}"
            f"_llm_model_fusion={llm_model_fusion}.json"
        )
        result_path = results_dir / result_filename

        # Skip if result already exists
        if result_path.exists():
            print(
                f"\033[93mSkipping combination {idx + 1}: Results already exist at {result_path}\033[0m\n"
            )
            continue

        # Run
        best_metrics = trainable(tunable_params, fixed_params, args)
        print_formatted_dict(best_metrics)

        ### * Storing results * ###

        # Add additional metadata to the metrics for traceability
        result_data = {
            "dataset": data_name,
            "model": model_name,
            "enable_text": enable_text,
            "TTF_module": TTF_module,
            "MMF_module": MMF_module,
            # "llm_model_fusion": llm_model_fusion,
            "metrics": best_metrics,
        }

        # Save to JSON
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=4)

        # Time tracking
        end_time_combination = time.time()
        elapsed_combination = end_time_combination - start_time_combination
        description = f"{data_name} | {model_name} | text={enable_text} | {TTF_module} | {MMF_module}"
        run_times.append((description, elapsed_combination))

        print(f"\033[92mTime taken: {elapsed_combination:.2f} seconds\033[0m\n")

    # Calculate total time
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    # PrettyTable summary
    table = PrettyTable()
    table.field_names = ["Combination", "Time (s)", "Percent of Total (%)"]

    for desc, t in run_times:
        percent = (t / total_elapsed_time) * 100
        table.add_row([desc, f"{t:.2f}", f"{percent:.2f}"])

    print("\n\033[96m===== Execution Time Summary =====\033[0m")
    print(table)
    print(
        f"\n\033[94mTotal time: {total_elapsed_time:.2f} seconds ({total_elapsed_time / 60:.2f} minutes)\033[0m"
    )

    print("### Done ###")
