"""
FSDP(完全分片数据并行)是PyTorch提供的一种先进的分布式训练技术，专门用于高效训练超大模型。
它通过智能分片模型参数、梯度和优化器状态，显著减少了每个GPU的内存占用，使训练超大模型成为可能。
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple


# 修改 preference_loss 函数以支持 per-sample beta
def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: Union[float, torch.FloatTensor],
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch.  支持 per-sample beta（向量）或标量 beta。
       返回：losses (per-sample), chosen_rewards (per-sample), rejected_rewards (per-sample)
    """
    device = policy_chosen_logps.device
    policy_rejected_logps = policy_rejected_logps.to(device)
    reference_chosen_logps = reference_chosen_logps.to(device)
    reference_rejected_logps = reference_rejected_logps.to(device)

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        # 明确为和 pi_logratios 同 shape 的 zeros（以便后续广播）
        ref_logratios = torch.zeros_like(pi_logratios)

    logits = pi_logratios - ref_logratios  # shape: (batch,)

    # --- 处理 beta：支持标量或 per-sample 向量（不要在此处强制 mean） ---
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    else:
        beta = beta.to(device=device, dtype=torch.float32)

    # 防止后续除零（尤其在 IPO 的 1/(2*beta)），给一个小下界
    eps = 1e-6
    if beta.dim() == 0:
        beta = torch.clamp(beta, min=eps)
    elif beta.dim() == 1:
        # 如果是一维向量，要求长度等于 batch 大小，或可广播到 logits
        if beta.shape[0] != logits.shape[0]:
            # 试图广播：如果长度为1则 expand；否则取前 batch 个并警告（保证可运行）
            if beta.numel() == 1:
                beta = beta.view(1).expand(logits.shape[0])
            else:
                beta = beta.view(-1)
                # 如果长于 batch，则截断；短于 batch 且不为1，则 expand 最后一个值
                if beta.shape[0] > logits.shape[0]:
                    beta = beta[:logits.shape[0]]
                else:
                    last = beta[-1].expand(logits.shape[0] - beta.shape[0])
                    beta = torch.cat([beta, last], dim=0)
        beta = torch.clamp(beta, min=eps)
    else:
        # 其他高维情况，尝试降维为一维可广播的向量，否则取第一个标量
        try:
            beta = beta.view(-1)
            if beta.shape[0] == 1:
                beta = beta.expand(logits.shape[0])
            elif beta.shape[0] != logits.shape[0]:
                beta = beta[:logits.shape[0]]
        except Exception:
            beta = torch.tensor(float(beta.view(-1)[0]), device=device, dtype=torch.float32)
            beta = torch.clamp(beta, min=eps)

    # 现在 beta 要么是标量张量（dim==0），要么与 logits 形状兼容（一维 batch）
    # 在计算时 broadcasting 会正常工作
    if ipo:
        # IPO 特殊形式，1/(2*beta) 要与 logits 做元素运算（注意 beta 不能为0）
        losses = (logits - 1.0 / (2.0 * beta)) ** 2
    else:
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.
        计算给定logits和labels下，每个序列的有效token的log概率总和（或平均）。
    Args:
        logits: Logits of the model (unnormalized).Shape: (batch_size, sequence_length, vocab_size)
        logits: 模型输出的未归一化对数概率。
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored.
        labels: 要计算对数概率的标签。值为 -100 的标签标记将被忽略。Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        average_log_prob: 布尔值，决定返回的是平均对数概率还是总对数概率。

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        形状为 (batch_size,) 的张量，包含给定对数下给定标签的平均/总对数概率。

    示例：
    由于logits是预测下一个token的概率分布，而labels是真实的下一个token，所以需要错位对齐：
    labels = labels[:, 1:].clone()    # 去掉第 1 个 token（起始 token）
    logits = logits[:, :-1, :]       # 去掉最后 1 个 token
    核心步骤：
    1. 对齐logits和labels（错位1个token）
    2. 计算loss_mask（标记哪些token需要计算loss）
    3. 计算每个token的对数概率（使用log_softmax + gather)
    4. 按average_log_prob返回平均或总和
    """
    # 断言检查，确保logits的前两个维度与labels的形状匹配
    assert logits.shape[:-1] == labels.shape

    # 对labels进行切片操作，去掉第一个token的标签（通常是起始token），使用clone()创建副本避免修改原始数据
    labels = labels[:, 1:].clone()
    # 对logits进行切片操作，去掉最后一个token的logits，使其与处理后的labels对齐
    logits = logits[:, :-1, :]
    # 创建掩码，标记哪些位置的标签不是-100（-100表示需要忽略的标签）
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    # 将标签中值未-100的位置置换为0（作为虚拟token，后续会通过掩码忽略这些位置）
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    """
    logits: 模型输出的原始预测值
    Shape: (batch_size, sequence_length, vocab_size)
    例如: (2, 5, 50000) 表示2条数据，每条5个token，词汇表大小50000
    labels: 真实的token ID（已处理忽略位）
    Shape: (batch_size, sequence_length)
    例如: (2, 5)，其中-100表示需要忽略的位置
    
    # 假设某位置logits = [3.0, 1.0, 0.5]
    log_softmax = torch.log_softmax([3.0, 1.0, 0.5], dim=-1)
    # 结果 ≈ [-0.422, -2.422, -2.922] (概率取log后为负值)
    
    labels.unsqueeze(2)：在labels的维度2（从0开始计数）插入一个大小为1的维度
    (batch_size, seq_len) → (batch_size, seq_len, 1)
    
    .squeeze(2)移除维度2（大小为1的维度）：(batch_size, seq_len, 1) → (batch_size, seq_len)
    """


    # 计算每个token的对数概率：
    # 1.logits.log_softmax(-1) - 在词汇表维度上计算log softmax，得到归一化的对数概率
    # 2.label.unsqueeze(2) - 将labels增加一个维度，形状变为(batch_size, sequence_length, 1)
    # 3.torch.gather() - 沿着词汇表维度(dim=2)收集每个标签对应的对数概率
    # 4. squeeze(2) - 移除多余的维度，恢复为(batch_size, sequence_length)

    if average_log_prob:
        # 用掩码过滤掉需要忽略的位置，沿着序列维度求和，然后除以非掩码token的数量，得到平均对数概率
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        # 用掩码过滤掉需要忽略的位置，然后沿着序列维度求和，得到总对数概率
        return (per_token_logps * loss_mask).sum(-1)
    # 函数最终返回一个形状为(batch_size,)的张量，包含每个样本的平均或总对数概率。

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    函数文字字符串，说明功能是将chosen和rejected输入拼接成单个张量。
    这个函数用于将训练数据中的“chosen”和“rejected”样本拼接成一个批次，方便后续的偏好学习
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        batch: 一个字典类型的batch参数，包含List或LongTensor类型的数据
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        返回包含'concatenated_input_ids'的字典

    这个函数的主要作用是：
    1. 找到chosen和rejected样本的最大长度
    2. 将所有的chosen开头的字段填充到最大长度并重命名
    3. 将所有rejected开头的字段填充后拼接到对应的concatenated字段后面
    4. 返回包含所有concatenated字段的新batch
    """
    # 计算chosen和rejected输入的最大序列长度，用于后续的填充对齐
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}  # 初始化空字典，用于存储拼接后的结果
    for k in batch:
        # 只处理以‘chosen’开头且值为Tensor的项
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            # k.startswith('chosen')筛选以 'chosen' 开头的字段（如 chosen_input_ids, chosen_labels）确保只处理正例样本相关数据，避免误操作其他字段
            # isinstance(batch[k], torch.Tensor)检查字段值是否为PyTorch张量，只有张量需要填充，其他类型（如列表或标量）可能需不同处理
            pad_value = -100 if 'labels' in k else 0
            # 对于response和labels的填充值是不一样的：
            # 如果字段名包含 'labels'（如 chosen_labels），则填充值设为 -100
            # 否则（如 chosen_input_ids），填充值设为 0
            concatenated_key = k.replace('chosen', 'concatenated')  # 替换键名
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
            # 对当前chosen张量进行填充，使其长度达到max_length，将结果存入concatenated_batch

    for k in batch:
        # 再次遍历，这次处理以‘rejected’开头的Tensor
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
            # 将rejected张量填充到max_length后，与之前处理的chosen张量在批次维度（dim=0）上进行拼接，更新concatenated_batch中的对应键
            # torch.cat是PyTorch 中的一个函数，用于将多个张量在指定维度上拼接起来。它接受一个张量序列作为输入，并沿着指定的维度将它们连接成一个新的张量。
    print("concatenated_batch:", concatenated_batch)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.

           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
           语言模型训练器，支持 SFT 或 DPO 训练。

            如果存在多个 GPU，则直接将模型拆分到多个 GPU 上，从而有效地提供 N 倍的可用内存，但不进行任何并行计算。
            关键点说明：
            1. 多GPU支持：通过ranl和world_size参数支持多GPU，但只是简单拆分模型到不同GPU上增加内存，不进行并行计算
            2. 两种训练模式：通过检查config.loss.name判断是SFT还是DPO训练
            3. 数据加载：使用get_batch_iterator统一处理训练和评估数据
            4. rank0_print：确保多进程环境下只打印一次信息
        """
        self.seed = seed  # 随机种子
        self.rank = rank  # 当前进程的rank（分布式训练用）
        self.world_size = world_size  # 总进程数（分布式训练用）
        self.config = config  # 训练配置（DictConfig）
        self.run_dir = run_dir  # 运行目录路径

        # 添加自适应beta参数
        self.adaptive_beta_config = {
            'beta_0': config.loss.beta,  # 基础beta值
            'lambda': config.loss.adaptive_lambda,  # λ参数
        }
        rank0_print(f"Adaptive beta config: {self.adaptive_beta_config}")

        # 获取tokenizer的名称或路径，优先使用config中指定的name_path，如果没有则使用模型名称或路径
        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        # 只有rank 0进程打印加载tokenizer的信息（避免多进程重复打印）
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        # 使用transformers的AutoTokenizer加载 tokenizer，指定缓存目录get_local_dir获得
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            # 如果没有pad_token_id，则使用eos_token_id作为pad_token_id
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 获得数据迭代器的参数
        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=False,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )
        # nqmes:数据集名称
        # shuffle:是否打乱数据
        # max_length:最大长度
        # max_prompt_length:最大prompt长度
        # 是否为SFT模式（通过检查config.loss.name）

        self.policy = policy  # 保存策略模型（policy）
        self.reference_model = reference_model  # 保留参考模型（reference_model）

        # 获取训练数据迭代器
        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        # 只有rank 0进程打印训练数据加载完成信息
        rank0_print(f'Loaded train data iterator')

        # 获取评估数据迭代器
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        # 将评估迭代器转换为列表形式保存
        self.eval_batches = list(self.eval_iterator)
        rank0_print("len(self.eval_batches): ", len(self.eval_batches))

        # 只有rank 0进程打印评估数据信息
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')
        """
        数据迭代器的作用：
        1.核心功能：
            动态生成训练/评估所需的批次数据（batches）
            处理文本数据的tokenization、截断、填充等预处理
            支持数据打乱和分布式训练
        2.特殊处理：
            区分prompt和response部分（通过max_prompt_length控制）
            适配不同训练模式，如SFT模式下可能不需要分离prompt/response
        3.设计动机
            动态加载，避免一次性加载全部数据
            分布式适配：通过rank！=0抑制非主进程的日志输出
            灵活性：通过cache_dir支持数据缓存加速后续训练
        """

    # 在 BasicTrainer 类中添加 compute_adaptive_beta 方法
    def compute_adaptive_beta(self, difficulty_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample adaptive beta:
            beta_i = beta_0 * exp(-lambda * D_i)
        输入 difficulty_scores: shape (batch,) 或 (N,)
        返回：torch.Tensor，shape 与 difficulty_scores 相同（per-sample β）
        """
        beta_0 = self.adaptive_beta_config['beta_0']
        lambda_val = self.adaptive_beta_config['lambda']

        # 把 difficulty_scores 放到正确设备 / dtype
        device = difficulty_scores.device if isinstance(difficulty_scores, torch.Tensor) else torch.device('cpu')
        dtype = difficulty_scores.dtype if isinstance(difficulty_scores, torch.Tensor) else torch.float32
        D = difficulty_scores.to(device=device, dtype=torch.float32)

        # 确保 beta_0 和 lambda_val 都是 scalar tensor（可以广播）
        if not isinstance(beta_0, torch.Tensor):
            beta_0_t = torch.tensor(beta_0, device=device, dtype=torch.float32)
        else:
            beta_0_t = beta_0.to(device=device, dtype=torch.float32)
            if beta_0_t.dim() > 0:
                beta_0_t = beta_0_t.mean()

        if not isinstance(lambda_val, torch.Tensor):
            lambda_t = torch.tensor(lambda_val, device=device, dtype=torch.float32)
        else:
            lambda_t = lambda_val.to(device=device, dtype=torch.float32)
            if lambda_t.dim() > 0:
                lambda_t = lambda_t.mean()

        # per-sample beta (broadcasting): shape = D.shape
        adaptive_beta = beta_0_t * torch.exp(-lambda_t * D)
        rank0_print("adaptive_beta: ", adaptive_beta)

        return adaptive_beta

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        # 接收一个包含输入数据的batch字典，返回一个元组，包含策略模型生成的文本和参考模型生成的文本
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""
        # 根据给定批次的输入，从策略（和参考模型，如果进行 DPO 训练）生成样本。
        """该方法主要完成以下工作：
        1. 使用策略模型生成文本样本
        2. 如果是DPO/IPO训练，也使用参考模型生成文本样本
        3. 处理多GPU情况下的输出收集
        4. 将生成的token ids解码为可读文本
        5. 返回两种模型的生成结果
        """

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[
        torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together."""
        concatenated_batch = concatenated_inputs(batch)

        # 确保输入数据在模型所在的设备上
        device = next(model.parameters()).device

        # 确保所有张量都在正确设备上
        for key in concatenated_batch:
            if torch.is_tensor(concatenated_batch[key]):
                concatenated_batch[key] = concatenated_batch[key].to(device)

        all_logits = model(
            concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask']
        ).logits.to(torch.float32)

        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch['concatenated_labels'],
            average_log_prob=False
        )

        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]

        return chosen_logps, rejected_logps

    # 修改 BasicTrainer 类中的 get_batch_metrics 方法
    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""
        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name in {'dpo', 'ipo'}:
            # --- 1) 从 batch 取 difficulty -> 构造 per-sample adaptive_beta ---
            if 'difficulty' in batch:
                device = next(self.policy.parameters()).device
                if isinstance(batch['difficulty'], list):
                    difficulty_scores = torch.tensor(batch['difficulty'], dtype=torch.float32, device=device)
                else:
                    difficulty_scores = batch['difficulty'].to(device=device, dtype=torch.float32)

                # compute_adaptive_beta 返回的是 per-sample beta（shape: batch）
                adaptive_beta = self.compute_adaptive_beta(difficulty_scores)

                # 记录到 metrics：保留 per-sample 和 mean 两种形式便于观测
                try:
                    metrics['adaptive_beta'] = adaptive_beta.detach().cpu().numpy().tolist()
                except Exception:
                    # 如果是标量则放入单元素列表
                    metrics['adaptive_beta'] = [float(adaptive_beta)]
                metrics['adaptive_beta_mean'] = float(adaptive_beta.mean().item())
            else:
                # 没有 difficulty 的情况下，使用配置中的 beta（标量）
                adaptive_beta = loss_config.beta
                metrics['adaptive_beta'] = [adaptive_beta]
                metrics['adaptive_beta_mean'] = float(adaptive_beta)

            # 接下来正常得到模型的 logps
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model,
                                                                                             batch)

            # --- 2) 确保 adaptive_beta 在正确 device / dtype 并与 logps 形状兼容 ---
            device = policy_chosen_logps.device
            if not isinstance(adaptive_beta, torch.Tensor):
                adaptive_beta = torch.tensor(adaptive_beta, device=device, dtype=torch.float32)
            else:
                adaptive_beta = adaptive_beta.to(device=device, dtype=torch.float32)

            # 如果是标量则 expand 到 batch shape；如果是一维且长度等于 batch 则保留
            if adaptive_beta.dim() == 0:
                adaptive_beta = adaptive_beta.expand(policy_chosen_logps.shape[0])
            elif adaptive_beta.dim() == 1 and adaptive_beta.shape[0] == policy_chosen_logps.shape[0]:
                pass
            else:
                # 其他形状尝试降维或取第一个元素作为 fallback
                adaptive_beta = adaptive_beta.view(-1)
                if adaptive_beta.shape[0] == 1:
                    adaptive_beta = adaptive_beta.expand(policy_chosen_logps.shape[0])
                else:
                    adaptive_beta = adaptive_beta[:policy_chosen_logps.shape[0]]

            # --- 3) 构造 loss_kwargs 并计算 loss（per-sample beta 会被传入 preference_loss） ---
            if loss_config.name == 'dpo':
                loss_kwargs = {
                    'beta': adaptive_beta,
                    'reference_free': loss_config.reference_free,
                    'label_smoothing': loss_config.label_smoothing,
                    'ipo': False
                }
                rank0_print(f"Using adaptive beta (mean): {metrics.get('adaptive_beta_mean')}")
            elif loss_config.name == 'ipo':
                loss_kwargs = {
                    'beta': adaptive_beta,
                    'ipo': True
                }
            else:
                raise ValueError(f'unknown loss {loss_config.name}')

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                **loss_kwargs)


            # 计算奖励正确率（优选样本奖励大于拒绝样本的情况）
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            # 多GPU情况下收集所有设备上的奖励值和准确率
            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            # 将奖励相关指标存入metrics字典
            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            # 收集并存储拒绝样本的对数概率
            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            # 处理SFT（监督微调）情况
            rank0_print("==========================处理SFT（监督微调）情况==========================")
            # 计算优选样本的logits
            policy_chosen_logits = self.policy(batch['chosen_input_ids'],
                                               attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            # 计算优选样本的对数概率
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)
            # SFT损失直接取对数概率的负值
            losses = -policy_chosen_logps

        # 收集并存储优选样本的对数概率
        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()
        # 收集并存储损失值
        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        # 返回平均损失和指标字典
        return losses.mean(), metrics

    # 修改 BasicTrainer 类中的 train 方法
    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        # 打印使用的优化器
        rank0_print(f'Using {self.config.optimizer} optimizer')
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.lr,  # 5e-7
            betas=(0.9, 0.999),  # 论文明确参数
            eps=1e-8,  # 论文明确参数
            weight_decay=0.0,  # 论文明确不使用权重衰减
            foreach=False  # 禁用foreach实现减少峰值内存
        )

        # 初始化学习率调度器（带warmup）
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        # --- 新增：加载优化器和调度器状态 ---
        if self.config.model.resume_from_checkpoint_policy:
            optimizer_state_dict = torch.load(os.path.join(self.config.model.resume_dir_policy, 'optimizer.pt'),
                                              map_location='cpu')
            self.optimizer.load_state_dict(optimizer_state_dict['state'])

            scheduler_state_dict = torch.load(os.path.join(self.config.model.resume_dir_policy, 'scheduler.pt'),
                                              map_location='cpu')
            self.scheduler.load_state_dict(scheduler_state_dict['state'])

            self.example_counter = scheduler_state_dict['step_idx']  # 恢复训练步数
            print(f"Resumed training from step {self.example_counter}")

        # 设置随机种子
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # 如果是DPO训练，设置参考模型为eval()
        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()

        # 初始化计数器
        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        # 训练主循环 - 修复循环结构
        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            #### 评估阶段 ####
            if self.example_counter % self.config.eval_every == 0 and (
                    self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                # 设置模型为评估模式
                self.policy.eval()

                # 初始化评估指标收集器
                all_eval_metrics = defaultdict(list)

                # 遍历评估批次
                for eval_batch in (
                tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    # 数据切片和移动
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size,
                                                                       self.rank)
                    # 计算评估指标
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    # 收集指标 - 修改这里
                    for k, v in eval_metrics.items():
                        if isinstance(v, list):
                            all_eval_metrics[k].extend(v)
                        else:
                            # 对于标量值（如adaptive_beta），使用append而不是extend
                            all_eval_metrics[k].append(v)

                # 计算平均评估指标并打印
                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')

                # 记录到wandb
                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                if self.example_counter > 0:
                    # 保存检查点
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
                #### END EVALUATION ####

            #### BEGIN TRAINING ####
            #### 训练阶段 ####
            rank0_print('========================进入训练阶段========================')
            self.policy.train()

            start_time = time.time()
            rank0_print('start_time: ', start_time)
            batch_metrics = defaultdict(list)

            # 梯度累计
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx,
                                                                    self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size,
                                                                   self.rank)

                # 计算损失和指标
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                # 反向传播（按累积步数缩放）
                (loss / self.config.gradient_accumulation_steps).backward()

                # 收集指标 - 修改这里
                for k, v in metrics.items():
                    if isinstance(v, list):
                        batch_metrics[k].extend(v)
                    else:
                        # 对于标量值（如adaptive_beta），使用append而不是extend
                        batch_metrics[k].append(v)

            # 梯度裁剪和优化
            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # 更新计数器和计时
            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            # 日志记录
            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            rank0_print('=====================训练结束=======================')
            #### END TRAINING ####

        # 训练完成后保存最终模型
        self.save()
        print(f'FINISHED TRAINING on rank {self.rank}')

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        # 对非FSDP模型的参数进行梯度裁剪
        """修改前
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm).item()  # 返回裁剪后的梯度范数值（转换成Python标量）（用于监控）"""
        # 修改后（保持0.5阈值但添加调试输出）
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            max_norm=0.5,  # 论文使用0.5（3.4.2节）
            norm_type=2  # 标准L2范数
        )
        if self.rank == 0 and self.batch_counter % 10 == 0:
            print(f"[梯度监控] step {self.batch_counter}: {grad_norm:.4f}")
        return grad_norm.item()
        # self.policy.parameters() 策略模型的所有参数
        # self.config.max_grad_norm 配置中定义的最大梯度范围阈值，限制所有参数的梯度范数不超过这个值
        # 明确说明仅适用于非FSDP模型（FSDP有内置的梯度裁剪）

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """
        :param step: 当前训练步数
        :param state: 要保存的状态字典
        :param metrics: 评估指标
        :param filename: 保存文件名
        :param dir_name: 保存目录（可选）
        """
        """Write a checkpoint to disk."""
        if dir_name is None:
            # 如果未指定目录，使用LATEST作为默认目录名
            dir_name = os.path.join(self.run_dir, f'LATEST')
            print(f"Saving to run_dir: {self.run_dir}")  # 检查实际保存路径
            print(f"Current working directory: {os.getcwd()}")
            print(f"Absolute run_dir path: {os.path.abspath(self.run_dir)}")

        # 创建目录（如果不存在）
        os.makedirs(dir_name, exist_ok=True)

        # 拼接完整输出路径
        output_path = os.path.join(dir_name, filename)

        # 主进程打印保存信息
        rank0_print(f'writing checkpoint to {output_path}...')

        # 保存到文件，包含：
        # - 当前训练步数
        # - 模型/优化器状态
        # - 评估指标（可为空）
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""
        """
        此函数是一个模型训练过程中的关键保存函数，用于将训练状态持久化到磁盘，支持断点续训和模型部署。
        执行save("checkpoints/strp-1000")后会生成：
        checkpoints/
          └── step-1000/
              ├── policy.pt       # 模型权重参数
              ├── optimizer.pt    # 优化器状态
              └── scheduler.pt    # 调度器状态/学习率调整历史
        
        每个.pt文件实际包含：
        {
            'step_idx': 1000,                 # 训练步数
            'state': {...},                   # 状态字典
            'metrics': {'loss': 0.123, ...}   # 评估指标
        }
        """

        # 1.保存策略模型，关键在于立即删除临时变量，对大模型训练尤为重要
        policy_state_dict = self.policy.state_dict()  # 获取模型状态
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)  # policy.pt为固定文件名
        del policy_state_dict  # 立即释放内存

        # 2.保存优化器状态，对于Adam优化器，会保存各参数的exp_avg和exp_avg_sq
        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        # 3.保存学习率调度器
        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        # write_state_dict方法内部可能使用临时文件+重命名操作，避免写入过程中崩溃导致文件损坏

        # 保存后立即验证
        saved_path = os.path.join(output_dir or os.path.join(self.run_dir, "LATEST"), "policy.pt")
        if os.path.exists(saved_path):
            print(f"✅ Successfully saved to {saved_path} (size: {os.path.getsize(saved_path)} bytes)")
        else:
            print(f"❌ File not found: {saved_path}")


class FSDPTrainer(BasicTrainer):
    # 此类是BasicTrainer的子类，专门用于使用PyTorch的FSDP进行分布式训练。
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.使用 PyTorch FSDP 将模型分片到多个 GPU 的训练器子类。
        
           This trainer will shard both the policy and reference model across all available GPUs.该训练器将在所有可用的 GPU 上对策略和参考模型进行分片。
           Models are sharded at the block level, where the block class name is provided in the config.模型在块级别进行分片，其中块类名称在配置中提供。
        """
        # 调用父类初始化
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)

        # 确保配置中指定了模型块名称
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        # 获取模型块类用于自动包装
        wrap_class = get_block_class_from_model(policy, config.model.block_name)

        # 创建自动包装策略
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},)

        # 共享的FSDP配置参数
        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,    # 自动包装策略
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片策略
            cpu_offload=CPUOffload(offload_params=False),   # 不卸载参数到GPU
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,    # 后向预取
            device_id=rank, # 设备ID
            ignored_modules=None,   # 不忽略任何模块
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        # 初始化策略模型FSDP包装
        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype,   # 参数精度
            reduce_dtype=mp_dtype,  # 规约精度
            buffer_dtype=mp_dtype   # 缓冲区精度
        )
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        # 激活检查点配置
        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                # 尝试导入FSDP激活检查点相关组件
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                # 创建非重入检查点包装器
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                # 应用激活检查点到策略模型
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        # 初始化参考模型FSDP包装（如果是DPO训练）
        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
        
        print('Loaded model on rank', rank)
        dist.barrier()  # 同步所有进程

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        # 使用FSDP内置的梯度裁剪方法
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        """Save model状态，只在rank 0进程保存"""
        # 1. 保存策略模型
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()  # 收集全量状态

        if self.rank == 0:  # 只在rank 0保存
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()  # 同步

        # 2.保存优化器状态
        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        # 3.保存调度器状态（不需要FSDP特殊处理）
        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()
        

class TensorParallelTrainer(BasicTrainer):
    # 这个类是BasicTrainer的子类，使用TensorParallel进行模型并行训练。
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """
        :param policy: 要训练的模型
        :param config: 训练配置
        :param seed: 随机种子
        :param run_dir: 运行目录
        :param reference_model:参考模型（用于DPO训练，可选）
        :param rank: 当前进程的rank（默认为0）
        :param world_size: 总进程数（默认为1）
        """
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.
           使用 TensorParallel 将模型分片到多个 GPU 的训练器子类。
           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        # 调用父类BasicTrainer的初始化方法
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)

        # 在主进程rank 0打印分片策略模型的信息
        rank0_print('Sharding policy...')
        # 使用tensor_parallel对策略模型进行分片，sharded=True表示使用分片模式
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            # 对参考模型进行分片，sharded=False表示不适用分片模式（可能因为参考模型只需要前向计算）
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        # 定义保存方法，接收以下参数：output_dir输出目录（可选），metrics：评估指标（可选）
        """Save (unsharded) policy state to disk.保存未分片的策略模型状态到磁盘"""
        with tp.save_tensor_parallel(self.policy):
            # 使用tensor_parallel的上下文管理器保存模型
            # 这会临时将分片模型合并为完整模型
            policy_state_dict = self.policy.state_dict()  # 获取策略模型的完整状态字典

        # 调用父类的write_state_dict方法保存模型状态，包含：当前的训练步数、模型状态、评估指标、文件名“policy.pt",输出目录
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict  # 删除状态字典释放内存

"""
关键点说明：
1.模型并行：
    使用tensor_parallel对模型进行分片
    策略模型使用sharded=True进行分片
    参考模型使用sharded=False（可能因为只需要前向计算）
2.保存机制：
    使用save_tensor_parallel上下文管理器临时合并分片模型
    保存完整的模型状态而非分片状态
    保存后立即释放内存
3.注意事项：
    文档中特别提到采样速度问题
    只在主进程打印日志信息(rank0_print)
这个实现专注于使用TrnsorParallel进行高效的模型并行训练，同时提供了方便的模型保护功能。
"""
        