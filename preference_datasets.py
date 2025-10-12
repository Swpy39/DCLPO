import datasets
import torch
from torch.utils.data import DataLoader, Dataset

from utils import rank0_print
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
from datasets import load_dataset
import json


def curri_dpo(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict]:
    print(f"Loading ultrafeedback_curriculum_dpo_pairs {split} dataset...")
    data_file = {
        "train": "./datasets/DiffCurri-DPO_dataset(α=1，γ=1).json",
        "test": "./datasets/DiffCurri-DPO_dataset(α=1，γ=1).json"
    }.get(split)
    # 暂时不需要使用test进行测试，而是训练完一个模型之后对不同模型的效果进行比较

    if not data_file:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")

    # 从接json中导出数据
    with open(data_file, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    # dataset = dataset[:100]  # 测试用
    print(f"Loaded {len(dataset)}")

    return dataset

# data = ultrafeedback_curri_dpo_pairs("train")
# print(data)

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None):
    if name == 'curri_dpo':
        data = curri_dpo(split, silent=silent, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    # 检查返回的数据是否包含必需的键('responses', 'pairs', 'sft_target')
    # 如果不符合，抛出AssertionError并显示实际存在的键
    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft', 'difficulty'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"
    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
        返回一个collate函数，用于将一批样本整理成模型可以接受的格式。主要功能时将一批样本（tokenized后的字典）整理成模型可接受的格式（填充到相同长度并转换为PyTorch Tensor）
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                # 对于以'_input_ids', '_attention_mask', '_labels'结尾的键，进行pad操作。
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    # 反转序列以实现左填充
                    # 原因：Transformer模型通常需要右对齐的prompt
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                # 填充值的选择
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # batch_first=True输出形状为(batch_size, seq_len)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])  # 反转恢复左填充
            else:
                # 对非Tensor字段处理：原始文本等直接保留，例如prompt、chosen
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
        处理一个样本（包含一个prompt，一个chosen回答，一个rejected回答）
         At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
        在此阶段，我们尚未转换为 PyTorch 张量；我们仅处理截断，以防提示 + 已选择或提示 + 已拒绝的响应过长。首先，我们截断提示；如果仍然过长，则截断已选择/已拒绝的响应。
         We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
        我们还为已选择/已拒绝的响应创建标签，其长度等于提示的长度与已选择/已拒绝的响应的长度之和，其中，提示标记的值为 -100。
    """

    # 移除数据中的EOS_token
    def remove_eos(tokens: Dict) -> Dict:
        """Remove EOS token if present in input_ids"""
        if tokenizer.eos_token_id in tokens['input_ids']:
            # Find all positions of EOS token
            eos_indices = [i for i, x in enumerate(tokens['input_ids']) if x == tokenizer.eos_token_id]
            # Remove EOS tokens and corresponding attention masks
            tokens['input_ids'] = [x for i, x in enumerate(tokens['input_ids']) if i not in eos_indices]
            tokens['attention_mask'] = [x for i, x in enumerate(tokens['attention_mask']) if i not in eos_indices]
        return tokens

    # 分别对prompt、chosen、rejected进行分词（不添加特殊token）
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    # Remove any existing EOS tokens
    prompt_tokens = remove_eos(prompt_tokens)
    chosen_tokens = remove_eos(chosen_tokens)
    rejected_tokens = remove_eos(rejected_tokens)

    # 确保分词结果中没有eos_token（因为后面会手动添加）
    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    # 在chosen和rejected分词结果后面添加eos_token。
    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    # 计算两个回答的最大长度
    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt 截断prompt
    # 如果prompt+回答的长度超过max_length，则根据truncation_mode（'keep_start'或'keep_end'）截断prompt到max_prompt_length。
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            # 保留开头，从开头开始直到长度达到要求
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            # 保留结尾，从后面开始确认，直到长度达到要求
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response 截断response
    # 如果仍然很长，则截断回答，前面是截断prompt到max_prompt_length
    # 此函数的目的是截断response到max_length-max_prompt_length
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    # 构建两个完整序列：prompt+chosen 和 prompt+rejected，并创建对应的labels（prompt部分为-100，回答部分为分词id）。
    # 这段代码的主要功能是构建用于监督学习的标签，具体是为偏好学习任务准备模型输入和对应的损失计算掩码
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]  # 复制input_ids
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    # prompt部分：全部设为-100，回答部分：保留原始token ID
    # chosen_input_ids = [1, 2, 3, 4, 5, 6]  # prompt + chosen
    # labels          = [-100, -100, -100, 4, 5, 6]  # 只计算回答部分的loss,避免prompt部分的干扰（因为prompt是输入，不需要预测）
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        # 外层循环：遍历三种序列类型（`chosen`, `rejected`, `prompt`）
        for type_key, tokens in toks.items():
            # 内层循环：遍历每种序列的分词字段（`input_ids`, `attention_mask`, `labels`）
            # 跳过token_type_ids（如果存在）
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens   # 例如：'chosen_input_ids', 'chosen_attention_mask'

    # 返回的`batch`字典包含以下类型的键：
    # - 原始文本键：`prompt`, `chosen`, `rejected`, `chosen_response_only`, `rejected_response_only`。
    # - 分词数据键：例如`chosen_input_ids`, `chosen_attention_mask`, `chosen_labels`, `rejected_input_ids`, `prompt_input_ids`等。
    return batch


# 修改数据训练方式
def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 32,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed: int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None) -> Iterator[Dict]:
    """
    此函数是一个生成批次数据的迭代器。
    参数：
    names:数据集名称列表（如['hh', 'curri_dpo']）
    tokenizer:HuggingFace Tokenizer实例
    split:数据划分('train', 'valid', 'test')
    batch_size:批大小
    shuffle:是否在每个epoch后打乱数据
    max_length:prompt+response的最大token长度
    max_prompt_length:prompt单独的最大token长度
    sft_mode:是否启用SFT模式
    n_epochs/n_examples:最大迭代轮次或样本数（二选一）
    seed:随机种子
    silent:是否关闭进度条
    cache_dir:数据集缓存目录
    """
    # 验证必须指定n_epochs或n_examples中的一个
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        # 如果silent为True，关闭进度条并只显示错误日志
        datasets.logging.disable_progress_bar()  # 禁用进度条显示
        datasets.logging.set_verbosity_error()  # 只显示错误日志

    with TemporarilySeededRandom(seed):
        # 上下文管理器，临时修改随机种子（保证线程安全）
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        # 预生成100万个随机种子，避免重复调用随机数生成器,确保实验可复现性,每个epoch使用不同种子实现动态shuffle
        flat_data = []  # 初始化扁平化数据列表
        for name in names:
            # 依次遍历每个在列表中的数据集
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'  # HH数据集用 keep_end（保留对话结尾）
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir).items():
                flat_data.append((prompt, data['responses'], data['pairs'], data['sft'], data['difficulty'], truncation_mode))
                # 数据扁平化：将所有数据集合并为统一结构的列表
                # 结构：[(prompt_str, [response1, ...], [(win_idx, lose_idx), ...], sft_target_str, trunc_mode),...]

    # 获取动态填充函数，处理不同长度的序列，实现prompt的左填充和response的右填充
    collate_fn = get_collate_fn(tokenizer)

    example_idx = 0  # 已处理样本计数
    batch = []  # 初始化当前batch
    done = False  # 终止标志

    # 加载复杂度文件，使得样本可以按照顺序进行输入
    file_path = 'dataset/difficulty_scores(α=1，γ=1).json'
    with open(file_path, 'r', encoding='utf-8') as file:
        difficulty_json = json.load(file)
    difficulty_length = len(difficulty_json)
    current_difficulty_num = 0

    while True:
        if done:
            break
        difficulty_score = difficulty_json[current_difficulty_num]
        for prompt, responses, pairs, sft, difficulty, truncation_mode in flat_data:
            if done:
                break

            # 调用tokenizer_batch_element处理特定的偏好对
            for num in range(len(pairs)):
                if difficulty[num] == difficulty_score:
                    batch_element = tokenize_batch_element(prompt,
                                                           responses[pairs[num][0]],  # chosen响应
                                                           responses[pairs[num][1]],  # rejected响应
                                                           truncation_mode,
                                                           tokenizer,
                                                           max_length,
                                                           max_prompt_length)
                    # 添加难度分数到batch_element
                    batch_element['difficulty'] = difficulty_score

                    batch.append(batch_element)
                    example_idx += 1
                    current_difficulty_num += 1  # 每找到一个样本便将数字加1
                    rank0_print("当前数据处理进程： ", current_difficulty_num)
                    rank0_print("当前的难度分数： ", difficulty_score)

        if current_difficulty_num == difficulty_length:
            rank0_print("数据处理完成！")
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(batch)

            for i in range(0, len(batch), batch_size):
                mini_batch = batch[i:i + batch_size]
                if len(mini_batch) == 0:  # 新增检查
                    continue
                collated = collate_fn(mini_batch)
                yield collated
            done = True
    print(f'FINISHED EXAMPLES on {split} split')
    print(f"最终{split}样本数量： ", example_idx)


# 严格匹配（除了空格之外的其他词）
def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    i = j = 0
    while i < len(str_a) and j < len(str_b):
        if str_a[i] == ' ':
            i += 1
            continue
        if str_b[j] == ' ':
            j += 1
            continue
        if str_a[i] != str_b[j]:
            return False
        i += 1
        j += 1

    # 检查剩余字符是否均为空格
    while i < len(str_a) and str_a[i] == ' ':
        i += 1
    while j < len(str_b) and str_b[j] == ' ':
        j += 1

    return i == len(str_a) and j == len(str_b)