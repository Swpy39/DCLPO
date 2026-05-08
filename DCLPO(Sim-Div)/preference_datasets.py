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
from omegaconf import OmegaConf

config = OmegaConf.load('./config/config.yaml')
cmd_overrides = OmegaConf.from_cli()
config = OmegaConf.merge(config, cmd_overrides)

def curri_dpo(split: str, silent: bool = False, config=config, cache_dir: str = None) -> Dict[str, Dict]:
    print(f"Loading HelpSteer_curriculum_dpo_pairs {split} dataset...")
    data_file = {
        "train": "./dataset/{}/HelpSteer(w1={},w2={}).json".format(config.curriculum_type, config.w1, config.w2),
        "test": "./dataset/{}/HelpSteer(w1={},w2={}).json".format(config.curriculum_type, config.w1, config.w2)
    }.get(split)

    if not data_file:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")

    with open(data_file, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    print(f"Loaded {len(dataset)}")

    return dataset

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None):
    if name == 'curri_dpo':
        data = curri_dpo(split, silent=silent, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft', 'difficulty'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"
    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """
    Tokenize a single batch element.
    """
    def remove_eos(tokens: Dict) -> Dict:
        """Remove EOS token if present in input_ids"""
        if tokenizer.eos_token_id in tokens['input_ids']:
            # Find all positions of EOS token
            eos_indices = [i for i, x in enumerate(tokens['input_ids']) if x == tokenizer.eos_token_id]
            # Remove EOS tokens and corresponding attention masks
            tokens['input_ids'] = [x for i, x in enumerate(tokens['input_ids']) if i not in eos_indices]
            tokens['attention_mask'] = [x for i, x in enumerate(tokens['attention_mask']) if i not in eos_indices]
        return tokens

    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    # Remove any existing EOS tokens
    prompt_tokens = remove_eos(prompt_tokens)
    chosen_tokens = remove_eos(chosen_tokens)
    rejected_tokens = remove_eos(rejected_tokens)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]  # 复制input_ids
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


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
                       cache_dir: Optional[str] = None,
                       config=config) -> Iterator[Dict]:

    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:

        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir).items():
                flat_data.append((prompt, data['responses'], data['pairs'], data['sft'], data['difficulty'], truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    example_idx = 0
    batch = []
    done = False

    file_path = './dataset/{}/HelpSteer_score_list(w1={},w2={}).json'.format(config.curriculum_type, config.w1, config.w2)
        difficulty_json = json.load(file)
    difficulty_length = len(difficulty_json)
    current_difficulty_num = 0

    while True:
        if done:
            break
        for prompt, responses, pairs, sft, difficulty, truncation_mode in flat_data:
            if done:
                break

            for num in range(len(pairs)):
                if done:
                    break
                difficulty_score = difficulty_json[current_difficulty_num]
                
                if difficulty[num] == difficulty_score:
                    batch_element = tokenize_batch_element(prompt,
                                                           responses[pairs[num][0]],  # chosen
                                                           responses[pairs[num][1]],  # rejected
                                                           truncation_mode,
                                                           tokenizer,
                                                           max_length,
                                                           max_prompt_length)
                    batch_element['difficulty'] = difficulty_score

                    batch.append(batch_element)
                    example_idx += 1
                    current_difficulty_num += 1
                    rank0_print("Number: ", current_difficulty_num)
                    rank0_print("Difficulty_score: ", difficulty_score)

                if current_difficulty_num == difficulty_length:
                    rank0_print("Finish！")
                    for i in range(0, len(batch), batch_size):
                        mini_batch = batch[i:i + batch_size]
                        if len(mini_batch) == 0:
                            continue
                        collated = collate_fn(mini_batch)
                        yield collated
                    done = True
    print(f'FINISHED EXAMPLES on {split} split')
    print(f"Number_{split}: ", example_idx)

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

    while i < len(str_a) and str_a[i] == ' ':
        i += 1
    while j < len(str_b) and str_b[j] == ' ':
        j += 1

    return i == len(str_a) and j == len(str_b)
