# MiniMind `trainer/train_*.py` 横向对比

- 环境公共逻辑：八个脚本都用 `init_distributed_mode` + `DistributedSampler`（可选）、`GradScaler`、`get_lr` 的线性 warmup-decay、`SkipBatchSampler` 续训、混合精度、周期性 `lm_checkpoint`。
- 主要差异集中在训练目标（预训练 / SFT / 蒸馏 / RLHF）、损失函数、数据形态（token-level vs. 对话/偏好样本）、生成和奖励策略。

## 速览表

| 脚本 | 任务/数据 | 损失与目标 | 关键流程/特性 |
| --- | --- | --- | --- |
| `train_pretrain.py` | 语言建模预训练，`PretrainDataset`，默认 `from_weight=none` | token 交叉熵（`reduction='none'` 手动 mask），`+aux_loss`（MoE），梯度累积=8 | 基础预训练，高 lr 5e-4，序列按 `loss_mask` 求均值 |
| `train_full_sft.py` | 全量 SFT，`SFTDataset`，默认从 `pretrain` 权重 | 与预训练相同的 token CE + `aux_loss` | 常规模型全参训练，lr 5e-7，梯度累积=1 |
| `train_lora.py` | LoRA SFT，`SFTDataset`，基于 `full_sft` 权重 | 同上 token CE + `aux_loss`，只反向 LoRA 参数 | 调用 `apply_lora` 冻结非 LoRA 参数；仅 LoRA 参数优化与裁剪；保存时只落盘 LoRA 权重 |
| `train_distillation.py` | 知识蒸馏，学生/教师 `SFTDataset` | 总损失 = `alpha * CE + (1-alpha) * KL(student||teacher)`；teacher logits 截断到学生 vocab | 教师 `eval+no_grad`，支持无教师退化为纯 CE；默认 alpha=0.5、T=1.5 |
| `train_distill_reason.py` | 推理蒸馏（无显式教师），`SFTDataset` 带 `<think>/<answer>` | token CE + `aux_loss`，对 `<think>`/`</think>`/`<answer>`/`</answer>` 位置将 `loss_mask` 权重放大 10× | 针对推理标签的重加权；默认数据 `r1_mix_1024`，权重名 `reason` |
| `train_dpo.py` | DPO 偏好对齐，`DPODataset`（chosen/rejected） | DPO 损失：`-logsigmoid(beta*(πΔ-βΔ_ref))`，序列级 log prob 按 mask 均值 | 参考模型冻结，策略模型更新；默认 beta=0.1，lr 4e-8 防遗忘 |
| `train_ppo.py` | PPO RLHF（Actor/Critic/Ref/Old-Actor），`RLAIFDataset` | `policy_loss`=PPO clip，`value_loss`=MSE(reward, value)，`+kl_coef*kl_ref` | 生成响应后计算奖励（含推理格式奖励可选），Critic 复用 LM 结构+value head，周期性同步 old_actor |
| `train_grpo.py` | GRPO（Group Relative），`RLAIFDataset` | per-token `-(exp(logp- stopgrad(logp))*adv - β*KL)`，batch 内按响应长度掩码均值 | 每 prompt 生成 `num_generations` 个响应做组内归一化优势；奖励=格式/标记奖励+reward model 打分 |
| `train_spo.py` | SPO（自博弈变体），`RLAIFDataset` | per-token `-logp*adv + β*KL`，优势基于自适应 baseline（AutoAdaptiveValueTracker） | 单样本生成；baseline 用 Beta 更新并可按 KL 自适应衰减；优势裁剪防爆；其余流程与 GRPO 类似 |

## 进一步要点

- **数据与序列长度**：预训练/SFT系列使用 `max_seq_len` 直接截断；RL 系列的 `max_seq_len` 作用于 prompt，生成长度由 `max_gen_len` 控制，总 `max_seq_len` 传入 `MiniMindConfig` 以覆盖 prompt+响应。
- **权重初始化**：预训练可 `none`，SFT 默认基于 `pretrain`，LoRA 基于 `full_sft`，推理蒸馏基于 `dpo`，RL 系列基于 `reason`（推理模式）或 `full_sft`。
- **梯度处理**：除 PPO/GRPO/SPO 的策略梯度直接 `loss.backward()` 外，其余均配合 `GradScaler`、梯度累积、`clip_grad_norm_`。LoRA 只裁剪/优化 LoRA 参数。
- **奖励设计（RL）**：
  - PPO/GRPO/SPO 共享的“格式/标记”奖励开关 `reasoning=1` 时，对 `<think>/<answer>` 标签结构加分。
  - PPO 使用 reward model 对整条回复（可选只对 `<answer>` 内容）打分；GRPO/SPO 在多/单样本生成后再加上 reward model 的分数。
  - SPO 引入自适应 baseline（`rho` 按 KL 或常数衰减）以稳定优势。
- **保存策略**：大多保存半精度 state_dict；LoRA 仅保存 LoRA 权重；PPO/GRPO/SPO 在 checkpoint 中额外保存调度器/critic 等状态；DPO/GRPO/SPO/PPO 参考模型/旧策略模型保持 `eval + no_grad`。
