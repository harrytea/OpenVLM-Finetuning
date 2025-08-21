import json
import os

# convert llava format to llama-factory format

# 显式指定要处理的 JSON 文件名列表
json_file_list = [
    "/opt/data/TGDoc_All_Data/TGDoc_All_Data/instruct/book/book.json",
    "/opt/data/TGDoc_All_Data/TGDoc_All_Data/instruct/grounding_finetune/finetune_grounding_new.json",
    "/opt/data/TGDoc_All_Data/TGDoc_All_Data/instruct/LLaVA-Instruct-150K/llava_instruct_150k.json",
    "/opt/data/TGDoc_All_Data/TGDoc_All_Data/instruct/LLaVAR-finetune/llavar_instruct_16k.json",
]

# 输出文件路径
output_json_path = 'merged_output.json'

def convert_back_format(input_json):
    """
    将单个 JSON 数据从目标格式转换为指定的格式。
    """
    output_json = []

    for item in input_json:
        # 处理 conversations -> messages
        messages = []
        for conversation in item.get("conversations", []):
            role = "user" if conversation["from"] == "human" else "assistant"
            content = conversation["value"]

            # 对 role 为 "user" 的内容进行处理，将 \n<image> 或 <image>\n 替换为 <image>
            if role == "user":
                content = content.replace("\n<image>", "<image>").replace("<image>\n", "<image>")

            # 添加 message 到列表
            messages.append({
                "content": content,
                "role": role
            })
        
        # 构建 "images" 属性，组合 image_folder 和 image
        image_folder = item.get("image_folder", "")
        image = item.get("image", "")
        full_image_path = os.path.join(image_folder, image)  # 拼接完整路径

        # 构建原始格式
        output_json.append({
            "messages": messages,
            "images": [full_image_path]  # 使用拼接后的路径放入 images 列表
        })
    
    return output_json

def process_file_list(file_list):
    """
    处理显式指定的文件列表并合并结果。
    """
    all_converted_data = []

    for file_name in file_list:
        print(f"正在处理文件: {file_name}")

        # 读取当前 JSON 文件
        with open(file_name, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # 转换数据格式并追加到总列表中
        converted_data = convert_back_format(input_data)
        all_converted_data.extend(converted_data)

    return all_converted_data

# 执行处理并保存结果
if __name__ == "__main__":
    # 处理文件列表
    merged_data = process_file_list(json_file_list)

    # 将合并后的数据保存到输出文件中
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"所有文件已处理并合并完成！结果已保存至 {output_json_path}")
