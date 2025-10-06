import os
import json
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# --- 1. 配置路径 ---
# 输入：您当前包含多个 safetensors 文件的模型目录
original_model_dir = Path("/data/250010046/verl-main/checkpoints/verl_grpo_vanilla_geo3k_No_KL_No_rectified/qwen2_5_vl_7b_function_rm/global_step_60/actor/merged_hf_model")

# 输出：一个新的目录，用于存放转换后的完整模型
converted_model_dir = Path("/data/250010046/verl-main/checkpoints/verl_grpo_vanilla_geo3k_No_KL_No_rectified/qwen2_5_vl_7b_function_rm/global_step_60/actor/converted_model_flat")
# --------------------


def convert_sharded_model(original_dir, converted_dir):
    """
    加载一个分片的 safetensors 模型，对所有分片和索引文件进行 key 的扁平化，
    并保存为一个新的模型目录。
    """
    if not original_dir.exists():
        print(f"错误：输入目录不存在于 '{original_dir}'")
        return

    # 创建输出目录，如果不存在的话
    print(f"创建输出目录: '{converted_dir}'")
    converted_dir.mkdir(parents=True, exist_ok=True)

    # 1. 查找所有的 safetensors 分片文件
    shard_files = sorted(list(original_dir.glob("*.safetensors")))
    if not shard_files:
        print(f"错误：在目录 '{original_dir}' 中没有找到 .safetensors 文件。")
        return
        
    print(f"找到 {len(shard_files)} 个 safetensors 分片文件，开始转换...")

    # 2. 循环处理每一个分片文件
    for shard_path in tqdm(shard_files, desc="Converting shards"):
        state_dict = load_file(shard_path)
        new_state_dict = {}
        
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("model.language_model."):
                new_key = "model." + key[len("model.language_model."):]
            elif key.startswith("model.visual."):
                new_key = key[len("model."):]
            
            new_state_dict[new_key] = value
            
        # 将转换后的 state_dict 保存到新目录，文件名保持一致
        output_shard_path = converted_dir / shard_path.name
        save_file(new_state_dict, output_shard_path, metadata={'format': 'pt'})

    print("所有 safetensors 分片转换完成！")

    # 3. 处理索引文件 (model.safetensors.index.json)
    index_path = original_dir / "model.safetensors.index.json"
    if index_path.exists():
        print("正在处理 index.json 文件...")
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # 创建新的 weight_map
        new_weight_map = {}
        original_weight_map = index_data.get("weight_map", {})
        
        for key, filename in original_weight_map.items():
            new_key = key
            if key.startswith("model.language_model."):
                new_key = "model." + key[len("model.language_model."):]
            elif key.startswith("model.visual."):
                new_key = key[len("model."):]
            
            new_weight_map[new_key] = filename
            
        # 更新索引数据中的 weight_map
        index_data["weight_map"] = new_weight_map
        
        # 将更新后的索引文件保存到新目录
        output_index_path = converted_dir / "model.safetensors.index.json"
        with open(output_index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        print("index.json 文件更新完成！")
    else:
        print(f"警告：在 '{original_dir}' 中没有找到 index.json 文件。")

    # 4. 拷贝其他所有非 safetensors 文件 (如 config.json, tokenizer.json 等)
    print("正在拷贝其他配置文件...")
    for file_path in original_dir.iterdir():
        if file_path.suffix != ".safetensors":
            # 使用 shutil.copy2 来保留文件元数据
            shutil.copy2(file_path, converted_dir / file_path.name)
            
    print("-" * 50)
    print("模型转换全部完成！")
    print(f"转换后的模型已保存至: {converted_dir}")
    print("您现在可以尝试使用这个新目录来加载模型。")


if __name__ == "__main__":
    convert_sharded_model(original_model_dir, converted_model_dir)