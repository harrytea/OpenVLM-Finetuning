# Qwen2-VL

### Environment

To avoid environmental issues, we strongly recommend using Dockerfile to run scripts

```shell
docker pull harrytea/qwen2-vl:v2
```

Versions of some important packages: accelerate==0.34.0, deepspeed==0.15.2, flash-attn==2.6.3, peft==0.11.1, transformers==4.45.0, torch==2.4.0, torchvision==0.19.0, torchaudio==2.4.0, cuda11.8

### Training

**Dataset**: follow [mllm_demo.json](data/mllm_demo.json) to prepare your data

**Note**: Remember to add the data card to [dataset_info.json](data/dataset_info.json) as follows

```
"mllm_demo": {
  "file_name": "mllm_demo.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "images": "images"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

**Train**: run `bash train.sh` to train your model, modify the corresponding config according your need


### Test

run the follow code

**Note**: maybe appear the following error `ValueError: No chat template is set for this processor. Please either set the chat_template attribute, or provide a chat template as an argument. See https://huggingface.co/docs/transformers/main/en/chat_templating for more information.`

Copy the `chat_template.json` file from the original `Qwen2-VL-7b-Instruct folder` to the finetuned folder to address the above issue

```python
import os

from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor


model = Qwen2VLForConditionalGeneration.from_pretrained("your_finetuned_model_ckpt", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("your_finetuned_model_ckpt")

query = "<image>Describe this image"
img_path = "1.jpg"
image = Image.open(img_path)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image",},
            {"type": "text", "text": f"{query}"},
        ],
    }
]
# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
inputs = inputs.to("cuda")

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=2048)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
answer = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(answer)
```