import torch
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32矩阵乘法加速
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource


# 为Hydra配置系统添加自定义解析器，动态生成运行目录路径
OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """
    Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer).
    分布式训练中每个工作进程的主函数
    :param rank: 当前进程排名
    :param world_size: 总进程数
    :param config: 训练配置
    :param policy: 待训练的策略模型
    :param reference_model: 参考模型（用于某些损失函数）
    :return:
    """
    if 'FSDP' in config.trainer:
        # 如果是FSDP训练器，初始化分布式设置
        init_distributed(rank, world_size, port=config.fsdp_port)  # 初始分布式环境
    
    if config.debug:
        # 调试模式时禁用wandb功能
        wandb.init = lambda *args, **kwargs: None  # 调试模式下禁用wandb初始化
        wandb.log = lambda *args, **kwargs: None  # 调试模式下禁用wandb日志

    if rank == 0 and config.wandb.enabled:
        # 在主进程且启用wandb时，设置并初始化wandb
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )  # 在主进程初始化wandb

    # 创建并启动指定类型的训练器
    TrainerClass = getattr(trainers, config.trainer)  # 根据配置获取训练器类
    print(f'Creating trainer on process {rank} with world size {world_size}')
    # 实例化训练器
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()  # 开始训练
    trainer.save()  # 保存模型


@hydra.main(version_base=None, config_path="config", config_name="config")  # Hydra装饰器指定配置路径和名称
# 主函数接收解析后的配置对象
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""
    # 训练的主入口点。验证配置、创建 / 初始化模型并启动工作进程
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)  # 解析配置中的变量引用

    missing_keys: Set[str] = OmegaConf.missing_keys(config)  # 验证配置完整性，检查缺失的配置项
    # 移除对新参数的缺失检查，因为它们可能有默认值
    if config.loss.name in {'dpo', 'ipo'}:
        # 设置自适应beta参数的默认值
        if not hasattr(config.loss, 'adaptive_lambda'):
            config.loss.adaptive_lambda = 0.1

    # 确保评估间隔是批次大小的整数倍
    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    # 为FSDP训练器设置默认端口
    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port  # 自动获取空闲端口

    # 打印并保存当前配置
    print(OmegaConf.to_yaml(config))
    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)  # 设置缓存目录
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}

    # 加载策略模型
    policy_dtype = getattr(torch, config.model.policy_dtype)  # 获取数据类型
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)  # 禁用dropout

    # --- 新增：加载已保存的策略模型权重（如果存在） ---
    if config.model.resume_from_checkpoint_policy:  # 添加一个配置项控制是否加载断点
        policy_state_dict = torch.load(os.path.join(config.model.resume_dir_policy, 'policy.pt'), map_location='cpu')
        policy.load_state_dict(policy_state_dict['state'])
        print(f"Loaded policy model from checkpoint at step {policy_state_dict['step_idx']}")

    # 加载参考模型（用于DPO/IPPO等特定损失函数）
    if config.loss.name in {'dpo', 'ipo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True,
            torch_dtype=reference_model_dtype, **model_kwargs)
        disable_dropout(reference_model)

        # --- 新增：加载已保存的参考模型权重（如果存在） ---
        if config.model.resume_from_checkpoint_ref:  # 添加一个配置项控制是否加载断点
            ref_state_dict = torch.load(os.path.join(config.model.resume_dir_ref, 'policy.pt'),
                                        map_location='cpu')
            reference_model.load_state_dict(ref_state_dict['state'])
            print(f"Loaded ref model from checkpoint at step {ref_state_dict['step_idx']}")
    else:
        reference_model = None

    # 加载预训练权重（如果指定）
    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo'}:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')
    
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()  # 获取GPU数量
        print('starting', world_size, 'processes for FSDP training')
        # 提高文件描述符限制（分布式训练需要）
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        # 启动多进程训练
        mp.spawn(worker_main, nprocs=world_size,
                 args=(world_size, config, policy, reference_model),
                 join=True)
    else:
        # 单进程训练
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()


"""
主要功能总结：
1.配置管理：使用Hydra进行灵活的配置管理
2.模型加载：
    · 本地加载预训练模型
    · 支持不同精度的模型（FP32/FP16/BF16)
    · 可选加载预训练权重
3.训练模式：
    · 支持单卡训练
    · 支持分布式训练FSDP
4.损失函数：
    · 支持DPO（直接偏好优化）
    · 支持IPO（迭代策略优化）
5.实验跟踪：
    · 集成Weights & Biases（wandb）进行实验跟踪
6.资源管理：
    · 自动设置缓存目录
    · 分布式训练时提高文件描述符限制
此文件用于训练基于Tranformer的大型语言模型，特别关注于偏好对齐技术如DPO。
"""

# export WANDB_MODE=offline  # 强制离线模式


