import subprocess
import os
import itertools
import shutil
from tqdm import tqdm

def create_args_combo(args_list):

    # Separate tuple keys and their values
    special_keys = [key for key in args_list if isinstance(key, tuple)]
    regular_keys = [key for key in args_list if not isinstance(key, tuple)]

    # Flatten the dictionary, keeping tuple keys together
    flat_args_list = {key: args_list[key] for key in regular_keys}
    for special_key in special_keys:
        flat_args_list.update({subkey: [value[i] for value in args_list[special_key]] for i, subkey in enumerate(special_key)})

    # Create all combinations of the regular argument values
    regular_combinations = [dict(zip(regular_keys, combination)) for combination in itertools.product(*(args_list[key] for key in regular_keys))]

    # Create final combinations by merging regular combinations with each tuple combination
    final_combinations = []
    for special_combination in args_list[special_keys[0]]:
        for regular_combination in regular_combinations:
            combined_dict = regular_combination.copy()
            combined_dict.update(zip(special_keys[0], special_combination))
            final_combinations.append(combined_dict)
    
    return final_combinations

def create_args_flag(args):
    res = []
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                res.append(f"--{k}")
        else:
            res.append(f"--{k}")
            res.append(str(v))
    return res

def run(path, args):
    folder_path, file_name = os.path.split(path)
    command = ["cd", folder_path, "&&", "python", file_name]
    command = command + create_args_flag(args)
    command = ' '.join(command)
    result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
    return result

if __name__ == "__main__":
    # ARCHITECTURE COMPETITION
    print("Length Competition !")
    
    ## LSTM
    path = "./LSTM/train.py"
    args_list = {
        ("train_data", "test_data") : [
            ("../dataset/podcast/long-text/train.txt", "../dataset/podcast/long-text/test.txt"),
        ],
        "seq_len" : [128, 256, 512, 1024],
        "d_model" : [256],
        "n_layer" : [3],
        "bidirectional" : [False],
        "gpu" : [0],
        "batch" : [8],
        "lr" : [3e-4],
        "epoch" : [10],
        "ckpt" : ["./model-experiment/model.ckpt"]
    }
    os.makedirs("./result/seq_len/LSTM", exist_ok=True)
    args_combo = create_args_combo(args_list)
    bar = tqdm(total=len(args_combo), desc="LSTM")
    for i, args in enumerate(args_combo):
        dataset_type = args["train_data"].split('/')[-2]
        run(path, args)
        shutil.copyfile("./LSTM/model-experiment/stats.json", f"./result/seq_len/LSTM/{args['seq_len']}.json")
        shutil.rmtree("./LSTM/model-experiment")
        bar.update()
    
    ## GPT
    path = "./MiniCharGPT/train.py"
    args_list = {
        ("train_data", "test_data") : [
            ("../dataset/podcast/long-text/train.txt", "../dataset/podcast/long-text/test.txt"),
        ],
        "seq_len" : [128, 256, 512, 1024],
        "d_model" : [512],
        "ff_dim" : [512],
        "n_head" : [1],
        "n_block" : [1],
        "gpu" : [0],
        "batch" : [8],
        "lr" : [3e-4],
        "epoch" : [10],
        "ckpt" : ["./model-experiment/model.ckpt"]
    }
    os.makedirs("./result/seq_len/GPT", exist_ok=True)
    args_combo = create_args_combo(args_list)
    bar = tqdm(total=len(args_combo), desc="GPT")
    for i, args in enumerate(args_combo):
        dataset_type = args["train_data"].split('/')[-2]
        run(path, args)
        shutil.copyfile("./MiniCharGPT/model-experiment/stats.json", f"./result/seq_len/GPT/{args['seq_len']}.json")
        shutil.rmtree("./MiniCharGPT/model-experiment")
        bar.update()