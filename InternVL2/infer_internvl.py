# CUDA_VISIBLE_DEVICES=2,3 python infer_internvl.py --dynamic --auto --checkpoint /llm-cfs-nj/person/harryyhwang/InternVL/internvl_chat/work_dirs/internvl_chat_v2_0/internvl2_8b_distance --prompt v2


import os
import os.path as osp

import sys
sys.path.append(os.getcwd())
sys.path.append("..")
sys.path.append("/llm-cfs-nj/person/harryyhwang/InternVL")
sys.path.append("/llm-cfs-nj/person/harryyhwang/InternVL/internvl_chat")

import json
from tqdm import tqdm
import fnmatch
import random
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import argparse
import math 
import argparse
import math
import re


import torch
from internvl_chat.internvl.model.internvl_chat import InternVLChatModel
from internvl_chat.internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pdb
from itertools import combinations


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def load_image(image_file, input_size=224):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size)
    if args.dynamic:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=args.max_num)
    else:
        images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument("--prompt", choices=["v0", "v1", "v2"], default="v1")
    args = parser.parse_args()


    # if args.auto: 
    #     os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # kwargs = {'device_map': 'auto'} if args.auto else {}
    kwargs = {
        "device_map": split_model("InternVL2-8B")
        # "device_map": split_model("InternVL2-26B")
        # "device_map": split_model("InternVL2-Llama3-76B")
    }

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, 
        load_in_4bit=args.load_in_4bit, 
        **kwargs
    ).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    generation_config = dict(
        do_sample=args.sample,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
    )


    query = "<image>\n" + "what is this"
    image_path = r"/llm-cfs-nj/person/harryyhwang/Qwen-VL/2.jpg"
    pixel_values = load_image(image_path, image_size).cuda().to(torch.bfloat16)

    while True:
        try:
            answer = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=query,
                generation_config=generation_config,
                verbose=True
            )         
            predicted_distance = extract_distance(answer)
            break
        except:
            print()
