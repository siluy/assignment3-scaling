import requests
import json
import time

API_URL = "http://hyperturing.stanford.edu:8000/loss"
API_KEY = "<YOUR_API_KEY_HERE>" # I'm not a stanford student...

# 用少的 FLOPS 探索最优 lr
lr_configs = [
    {"d_model": 768, "num_layers": 12, "num_heads": 12, "learning_rate": 1e-4, "train_flops": int(1e16)},
    {"d_model": 768, "num_layers": 12, "num_heads": 12, "learning_rate": 3e-4, "train_flops": int(1e16)},
    {"d_model": 768, "num_layers": 12, "num_heads": 12, "learning_rate": 8e-4, "train_flops": int(1e16)},
]

BEST_LEARNING_RATE = 3e-4

main_configs = [
    # C1 = 1e17 FLOPs 的 6 个模型
    {"d_model": 512, "num_layers": 6, "num_heads": 8, "train_flops": int(1e17)},
    {"d_model": 768, "num_layers": 8, "num_heads": 12, "train_flops": int(1e17)},
    {"d_model": 768, "num_layers": 12, "num_heads": 12, "train_flops": int(1e17)},
    {"d_model": 1024, "num_layers": 10, "num_heads": 16, "train_flops": int(1e17)},
    {"d_model": 1024, "num_layers": 16, "num_heads": 16, "train_flops": int(1e17)},
    {"d_model": 1024, "num_layers": 24, "num_heads": 16, "train_flops": int(1e17)},
    
    # C2 = 3e17 FLOPs 的 4 个模型
    {"d_model": 768, "num_layers": 8, "num_heads": 12, "train_flops": int(3e17)},
    {"d_model": 1024, "num_layers": 10, "num_heads": 16, "train_flops": int(3e17)},
    {"d_model": 1024, "num_layers": 16, "num_heads": 16, "train_flops": int(3e17)},
    {"d_model": 1024, "num_layers": 24, "num_heads": 16, "train_flops": int(3e17)},
]
for config in main_configs:
    config['learning_rate'] = BEST_LEARNING_RATE
    config['batch_size'] = 256

def run_experiment(config, results_list):
    params = config.copy()
    params['api_key'] = API_KEY
    print(f"正在进行实验：{config}")
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        result_record = config.copy()
        result_record['loss'] = data.get('loss')
        result_record['total_flops_used'] = data,get('total_flops_used')
        results_list.append(result_record)
        print(f"  -> 成功！Loss: {result_record['loss']:.4f}, "
              f"累计预算已使用: {result_record['total_flops_used']:.2e} FLOPs")
    
    except requests.exceptions.RequestException as e:
        print(f"  -> 实验失败: {e}")
        try:
            print(f"API 返回: {response.json().get('message')}")
        except:
            pass

def run_all_experiments(experiment_list, output_filename):
    results = []
    for config in experiment_list:
        run_experiment(config, results)
        time.sleep(1)
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

if __name__ == "__main__":
    run_all_experiments(lr_configs, 'lr_exploration_results.json')
    run_all_experiments(main_configs, 'main_experiment_results.json')
