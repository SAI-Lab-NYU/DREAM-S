import argparse
import base64
import os
import random
from io import BytesIO

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
)


def load_image(base64_string):
    base64_string = base64_string.replace("\n", "")
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


class MMTBenchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]
        image_base64 = record["image"]
        conversation = record["conversation"]
        image = load_image(image_base64)
        return image, conversation


def collate_fn(batch):
    images, conversations = zip(*batch)
    prompts = [processor.apply_chat_template(
        msg, add_generation_prompt=True) for msg in conversations]
    return prompts, list(images)


def compute_attention_entropy(attn_weights, eps=1e-8):
    with torch.no_grad():
        attn_weights = attn_weights
        entropy = - (attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
        attn_entropy = entropy.mean(dim=1)
        del entropy
        torch.cuda.empty_cache()
    return attn_entropy


def mid_attention_score(attn_tuple, best_layer_idx, eps=1e-8):
    L = len(attn_tuple)
    B, H, S, K = attn_tuple[0].shape

    score_list = []
    for l in range(L):
        max_per_head = attn_tuple[l].max(dim=-1).values    # (B, H, S)
        score_l = max_per_head.mean(dim=1)                 # (B, S)
        score_list.append(score_l)
    score_stack = torch.stack(score_list, dim=0).to('cpu') # (L, B, S)

    best_layer_score = torch.gather(
        score_stack,
        dim=0,
        index=best_layer_idx.unsqueeze(0)
    ).squeeze(0)                                           # (B, S)

    top_layer_score = score_stack[-1]                      # (B, S)

    return best_layer_score, top_layer_score


def mid_feature_collect_and_score(features_tuple, attn_tuple, eps=1e-8):
    L = int(0.98 * (len(features_tuple) - 1))
    B, s, d = features_tuple[0].shape
    features_stack = torch.stack(features_tuple[:L+1], dim=0).to("cpu")
    device = "cpu"

    att_entropy_list = []
    for l in range(L):
        att_entropy = compute_attention_entropy(attn_tuple[l], eps=eps).to(device)  # (B, qlen)
        att_entropy_list.append(att_entropy)
    att_entropy_stack = torch.stack(att_entropy_list, dim=0).to(features_stack.device)  # (L, B, s)

    total_metric = att_entropy_stack[1:L-1]  # (L-2, B, s)

    best_layer_idx = total_metric.argmin(dim=0) + 1  # (B, s)

    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)  # (1, B, s, 1)
    best_features_b = torch.gather(features_stack, dim=0,
                                   index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)  # (B, s, d)

    best_layer_scores, top_layer_scores = mid_attention_score(attn_tuple, best_layer_idx)

    return best_features_b.cpu(), best_layer_scores.cpu(), top_layer_scores.cpu()


def find_subsequence(tensor, subsequence):
    """Find the first occurrence of subsequence in tensor"""
    tensor_list = tensor.tolist()
    subseq_list = subsequence if isinstance(subsequence, list) else subsequence.tolist()

    for i in range(len(tensor_list) - len(subseq_list) + 1):
        if tensor_list[i:i+len(subseq_list)] == subseq_list:
            return i
    return -1


@torch.no_grad()
def ge(data):
    prompts, images = data
    if images is None:
        return None
    inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=5120,
        padding_side='left'
    )
    device = bigmodel.device
    inputs = inputs.to(device)
    seq_length = inputs.input_ids.shape[1]

    outs_big = bigmodel(**inputs, output_hidden_states=True, output_attentions=True)
    mid_feature, _, _ = mid_feature_collect_and_score(outs_big.hidden_states, outs_big.attentions)

    assist_position = find_subsequence(inputs.input_ids[0], assist_tokens)

    last_image_position = (inputs.input_ids[0] == 32000).nonzero()[-1, 0].item()
    image_start = 5
    image_end = last_image_position + 1

    text_end = assist_position + 5

    inputs_prompt = type(inputs)({
        'input_ids': inputs.input_ids[:, :text_end+1],
        'attention_mask': inputs.attention_mask[:, :text_end+1],
        'pixel_values': inputs.pixel_values,
        'image_sizes': inputs.image_sizes
    })

    try:
        outs_big_prompt = bigmodel(**inputs_prompt, output_hidden_states=True, output_attentions=True)
        _, _, target_score_prompt = mid_feature_collect_and_score(outs_big_prompt.hidden_states, outs_big_prompt.attentions)
    except ValueError as e:
        if "Image features and image tokens do not match" in str(e):
            print(f"Skipping sample due to image token mismatch: {e}")
            return None
        else:
            raise e
    except Exception as e:
        print(f"Unexpected error in model forward pass: {e}")
        return None

    image_attention_score = target_score_prompt[:, image_start:image_end].to(device)

    # Define pruning ratios from 0.1 to 1.0
    pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    pruning_indices_list = []
    for ratio in pruning_ratios:
        if ratio != 1:
            top_image_attention_rank_index = image_attention_score.topk(
                int((image_end - image_start) * ratio)
            ).indices + image_start
            top_image_attention_rank_index = top_image_attention_rank_index.squeeze(0)
            keep_indexs = torch.cat((
                torch.arange(image_start, device=device),
                top_image_attention_rank_index,
                torch.arange(image_end, seq_length, device=device)
            )).cpu()
            keep_indexs = keep_indexs.sort().values
            pruning_indices_list.append(keep_indexs)

    # Compute Loss Mask
    loss_mask = torch.zeros_like(inputs.input_ids)
    for i in range(inputs.input_ids.size(0)):
        tokens = inputs.input_ids[i]
        start_idx = None
        j = 0
        while j < tokens.size(0):
            if start_idx is None and j <= tokens.size(0) - assist_len and tokens[j:j+assist_len].tolist() == assist_tokens:
                start_idx = j
                j += assist_len
                continue
            if start_idx is not None and j <= tokens.size(0) - end_len and tokens[j:j+end_len].tolist() == end_tokens:
                loss_mask[i, start_idx+assist_len:j] = 1
                start_idx = None
                j += end_len
                continue
            j += 1

    td = {"loss_mask": loss_mask.cpu()}
    td["attention_mask"] = inputs.attention_mask.cpu()
    td["inputs_embeds"] = outs_big.hidden_states[0].cpu()
    td["hidden_state_mid_a"] = mid_feature.cpu()
    td["target"] = outs_big.hidden_states[-1].cpu()

    random_target_layer = random.choice([-1, -2, -3, -4, -5])
    td["target2"] = outs_big.hidden_states[random_target_layer].cpu()
    td["pruning_indices"] = pruning_indices_list
    td["pruning_ratios"] = pruning_ratios

    return td


def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length = len(os.listdir(name))
    idx = current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
    parser.add_argument('--outdir', type=str, default='outdir0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

    target_model_id = "/home/asperger/models/llava-v1.6-vicuna-7b-hf"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    bigmodel = LlavaNextForConditionalGeneration.from_pretrained(
        target_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=quantization_config
    )
    processor = AutoProcessor.from_pretrained(target_model_id)

    if "vicuna" in target_model_id:
        assist_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        end_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        image_tokens = processor.tokenizer.encode("<image>", add_special_tokens=False)
    elif "mistral" in target_model_id:
        assist_tokens = processor.tokenizer.encode("[/INST]:", add_special_tokens=False)
        end_tokens = processor.tokenizer.encode("[INST]:", add_special_tokens=False)
        image_tokens = processor.tokenizer.encode("<image>", add_special_tokens=False)

    assist_len = len(assist_tokens)
    end_len = len(end_tokens)

    ds = load_dataset("ge_data/processed_data",
                      data_files="ge_data/processed_data/train.jsonl", split="train")
    ds = ds.shuffle(seed=42)
    if len(ds) < args.end:
        args.end = len(ds)
    ds = ds.select(range(args.start, args.end))
    dataset = MMTBenchDataset(ds)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True
    )

    outdir = f'{args.outdir}/{args.index}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for data in tqdm(data_loader):
        torch.cuda.empty_cache()
        outdata = ge(data)
        if outdata is not None:
            writedata(outdir, outdata)