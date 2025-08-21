import os
import sys

sys.path.append(os.getcwd())
sys.path.append("..")

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image    
import pdb
import torch


class LLaVANexTHandler:
    def __init__(self):
        pass
    
    def initialize_llm(self, model_path, device="cuda"):
        # model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        # Use Flash-Attention 2 to further speed-up generation
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True)
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model.to(device)

        self.processor = processor
        self.model = model
    

    def generate(self, image, question):
        image = Image.open(image)
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cuda")



        while True:
            try:
                # pdb.set_trace()
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=128
                )
                answer = self.processor.decode(output[0], skip_special_tokens=True)
                answer = answer.split("ASSISTANT: ")[1]
                break
            except:
                print()
        return answer


if __name__ == '__main__':
    my_vlm = LLaVANexTHandler()
    my_vlm.initialize_llm(model_path="/opt/models/llava-hf/llava-hf/llava-v1.6-vicuna-7b-hf-instruct")
    answer = my_vlm.generate("/opt/llavanext/data/mllm_demo_data/1.jpg", "what is this")
    print(answer)
