from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 步骤 1: 加载基础模型和 tokenizer（使用非量化版本）
base_model_name = "meta-llama/Llama-2-7b-hf"  # 替换为你的基础模型
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 步骤 2: 加载 LoRA adapter
lora_adapter_path = "./lora_adapter"  # 替换为你的 adapter 路径
peft_model = PeftModel.from_pretrained(model, lora_adapter_path)

# 步骤 3: 合并 adapter 到基础模型
merged_model = peft_model.merge_and_unload()

# 步骤 4: 保存合并后的模型（可选）
output_dir = "./merged_model"
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# 现在 merged_model 就是一个独立的模型，可以直接用于推理
print("模型合并完成！")