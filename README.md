# HEK293T 条件下 Gluc 密码子优化流程说明

本项目用于在固定 Gluc 蛋白氨基酸序列的前提下，先构建同义 CDS 起始序列池，再使用强化学习在 HEK293T 细胞环境下进行同义密码子优化。整体流程分为两个主要步骤：

1. `step1_generate_gluc_random_cds.py`：均匀随机生成 Gluc 同义 CDS，并筛选最低 CAI 的 TopK 序列作为 RL 起始序列。
2. `step2_RL_final.py`：读取起始 CDS，使用冻结翻译预测模型 `best_model.p` 与 HEK293T 表达环境，对 CDS 进行 Actor-Critic 强化学习优化。

---

## 1. 文件说明

| 文件 | 作用 |
|---|---|
| `environment.yml` | Conda 环境配置文件，用于复现 `codonoptimizer` 运行环境。 |
| `step1_generate_gluc_random_cds.py` | 生成随机同义 CDS 池，并筛选最低 CAI 的 TopK 起始序列。 |
| `step2_RL_final.py` | 强化学习主程序，使用 `score.py-aligned` 打分逻辑优化 Gluc CDS。 |

---

## 2. 环境配置

### 2.1 使用 `environment.yml` 创建环境

```bash
conda env create -f environment.yml
conda activate codonoptimizer
```

如果环境已经存在，可以更新：

```bash
conda env update -n codonoptimizer -f environment.yml --prune
conda activate codonoptimizer
```

### 2.2 主要依赖

该环境基于 Python 3.8，主要依赖包括：

- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `ViennaRNA`
- `ribodecode`
- `translationmodel`
- `matplotlib`
- `biopython`

其中 `ViennaRNA` 用于计算 MFE；`torch` 用于加载翻译预测模型和训练 Actor-Critic 策略网络。

---

## 3. 项目所需数据和模型路径

`step2_RL_final.py` 中多个路径是写死的。正式运行前应确认以下文件存在：

| 类型 | 默认路径 | 说明 |
|---|---|---|
| 起始 CDS 文件 | `/data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026_top10_sequences` | RL 起始序列文件，一行一条 CDS。 |
| 翻译预测模型 | `/data/hyliu/HEK293T_RL/Models/best_model.p` | 冻结的 RiboDecode-style 翻译预测模型。 |
| 模型配置 | `/data/hyliu/HEK293T_RL/Models/model_config.json` | 模型结构配置文件。 |
| HEK293T 条件 | `/data/hyliu/HEK293T_RL/conditions/HEK293T_10552_RPKM.npz` | HEK293T 10552 维原始 RPKM 向量。 |
| CAI 权重 | `/data/hyliu/HEK293T_RL/metrics/hek293t_codon_weights.txt` | HEK293T 高表达背景密码子权重。 |
| CSI 权重 | `/data/hyliu/HEK293T_RL/metrics/human_csi_weights.txt` | Human overall CDS 背景权重。 |
| 高表达指标表 | `/data/hyliu/HEK293T_RL/metrics/HEK293T_full_high_expression_metrics.txt` | 用于 GMM 生物中心建模。 |
| MFE 表 | `/data/hyliu/HEK293T_RL/metrics/HEK293T_full_high_expression_CDS_with_MFE_only_cds_mfe.txt` | 用于长度校正 MFE residual。 |
| 输出目录 | `/data/hyliu/HEK293T_RL/RL_results_Gluc` | RL 结果输出目录。 |

---

## 4. Step 1：生成低 CAI 起始序列池

### 4.1 脚本功能

`step1_generate_gluc_random_cds.py` 的作用是：

1. 固定 Gluc 氨基酸序列；
2. 对每个氨基酸位置均匀随机选择一个同义密码子；
3. 生成 `N` 条唯一 CDS；
4. 计算每条 CDS 的 HEK293T CAI、GC 和 Rare fraction；
5. 按 CAI 从低到高排序；
6. 选择最低 CAI 的 TopK 序列作为 RL 起始序列；
7. 可选地只对 TopK 计算 MFE。

该步骤的设计是：

```text
unconditional uniform random synonymous generation
+ post-hoc lowest-CAI selection
```

也就是说，生成阶段不使用 CAI 权重偏置采样，而是在生成完成后再按 CAI 筛选。

### 4.2 小规模测试命令

建议先生成 1 万条测试：

```bash
cd /data/hyliu/HEK293T_RL

python step1_generate_gluc_random_cds.py \
  --use_default_gluc \
  --cai_weight_file /data/hyliu/HEK293T_RL/metrics/hek293t_codon_weights.txt \
  --out_dir /data/hyliu/HEK293T_RL/challenge_start_pool/test_run \
  --prefix Gluc_test_uniformRandom10000_lowest10_seed2026 \
  --n 10000 \
  --top_k 10 \
  --seed 2026 \
  --max_attempts 200000 \
  --rare_threshold 0.30 \
  --stop_mode balanced_three \
  --min_codon_diff 8 \
  --calc_mfe_for_selected
```

### 4.3 正式生成命令

如果要生成 1000 万条随机同义 CDS 并筛选 Top10：

```bash
cd /data/hyliu/HEK293T_RL

python step1_generate_gluc_random_cds.py \
  --use_default_gluc \
  --cai_weight_file /data/hyliu/HEK293T_RL/metrics/hek293t_codon_weights.txt \
  --out_dir /data/hyliu/HEK293T_RL/challenge_start_pool \
  --prefix Gluc_MF882921_uniformRandom_seed2026 \
  --n 10000000 \
  --top_k 10 \
  --seed 2026 \
  --max_attempts 20000000 \
  --rare_threshold 0.30 \
  --stop_mode balanced_three \
  --min_codon_diff 8 \
  --calc_mfe_for_selected
```

### 4.4 Step 1 输出文件

假设 `--prefix Gluc_MF882921_uniformRandom_seed2026`，输出文件为：

```text
/data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026.random_pool.tsv
/data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026.lowestCAI_top10.tsv
/data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026.lowestCAI_top10_sequences.txt
/data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026.summary.txt
```

其中最重要的是：

```text
Gluc_MF882921_uniformRandom_seed2026.lowestCAI_top10_sequences.txt
```

该文件一行一条 CDS，可作为 `step2_RL_final.py` 的起始序列输入。

### 4.5 重要路径提醒

当前 `step2_RL_final.py` 中的起始序列路径写死为：

```python
ORIGINAL_FILE = "/data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026_top10_sequences"
```

而 `step1` 默认输出的文件名格式是：

```text
<prefix>.lowestCAI_top10_sequences.txt
```

因此运行 Step 2 前，需要二选一：

**方法 A：复制并重命名 Step 1 输出文件**

```bash
cp /data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026.lowestCAI_top10_sequences.txt \
   /data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026_top10_sequences
```

**方法 B：修改 `step2_RL_final.py` 中的 `ORIGINAL_FILE`**

```python
ORIGINAL_FILE = "/data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026.lowestCAI_top10_sequences.txt"
```

---

## 5. Step 2：强化学习优化 CDS

### 5.1 脚本功能

`step2_RL_final.py` 是强化学习主程序。它从 Step 1 得到的低 CAI Gluc 起始 CDS 出发，在保持目标氨基酸序列不变的前提下，通过 Actor-Critic 进行同义密码子替换优化。

核心设计如下：

```text
状态：当前 CDS 序列 + 当前优化进度特征
动作：选择一个密码子位置，并替换为该氨基酸对应的另一个同义密码子
约束：目标蛋白氨基酸序列不变
主目标：score_z + CAI_z
最终筛选：s_bio >= CENTER_MIN
```

当前主目标为：

```python
objective = score_z + cai_z
```

其中：

- `score_z`：冻结翻译预测模型输出的 translation score 标准化值；
- `cai_z`：HEK293T CAI 标准化值；
- `s_bio`：基于 HEK293T 高表达序列特征分布的中心相似性，仅用于最终部署筛选；
- `CSI`：保留用于日志和结果报告，不参与主目标；
- `MFE`：以长度校正残差 `mfe_residual` 进入生物可行性建模。


## 6. Step 2 运行命令

### 6.1 单条起始序列测试

建议先测试第 1 条起始序列：

```bash
cd /data/hyliu/HEK293T_RL
conda activate codonoptimizer

export CUDA_VISIBLE_DEVICES=0
export RL_SINGLE_START_ID=1

python step2_RL_final.py
```

测试完成后取消单起点环境变量：

```bash
unset RL_SINGLE_START_ID
```

### 6.2 正式并行运行

如果有 4 张 GPU，可以让最多 4 个 worker 并行：

```bash
cd /data/hyliu/HEK293T_RL
conda activate codonoptimizer

export CUDA_VISIBLE_DEVICES=0,1,2,3
export RL_MAX_PARALLEL_START_JOBS=4

nohup python step2_RL_final.py > step2_RL_final.log 2>&1 &
```

查看主日志：

```bash
tail -f /data/hyliu/HEK293T_RL/step2_RL_final.log
```

查看并行 worker 日志：

```bash
tail -f /data/hyliu/HEK293T_RL/RL_results_Gluc/parallel_logs/start_01.log
```

如果只使用 1 张 GPU，可以设置：

```bash
export CUDA_VISIBLE_DEVICES=0
export RL_MAX_PARALLEL_START_JOBS=1
nohup python step2_RL_final.py > step2_RL_final.log 2>&1 &
```

---

## 7. Step 2 输出结果

默认输出目录为：

```text
/data/hyliu/HEK293T_RL/RL_results_Gluc
```

主要结果文件包括：

```text
RL_results_Gluc/
├── all_sequence_summary.txt
├── parallel_run_summary.json
├── parallel_logs/
│   ├── start_01.log
│   ├── start_02.log
│   └── ...
├── seq_0001/
│   ├── trace.txt
│   ├── episode_summary.txt
│   ├── final_candidates.txt
│   ├── final_top10.txt
│   ├── run_result_summary.json
│   └── run_result_summary.txt
├── seq_0002/
│   └── ...
└── ...
```

文件说明：

| 文件 | 含义 |
|---|---|
| `trace.txt` | 每个 episode/step 的动态优化轨迹，包括 score、CAI、reward、替换位置等。 |
| `episode_summary.txt` | 每个 episode 的汇总结果。 |
| `final_candidates.txt` | 最终候选池，通常包含更多候选。 |
| `final_top10.txt` | 当前起始序列对应的最终 Top10 候选。 |
| `run_result_summary.json` | 单个起始序列 run 的完整 JSON 汇总。 |
| `run_result_summary.txt` | 单个起始序列 run 的表格版汇总。 |
| `all_sequence_summary.txt` | 所有起始序列的总汇总。 |
| `parallel_run_summary.json` | 并行任务完成情况，包括 completed / failed / log_dir。 |

正式分析时建议重点查看：

```text
seq_*/final_top10.txt
seq_*/final_candidates.txt
all_sequence_summary.txt
parallel_run_summary.json
```

---

## 8. 结果解释

### 8.1 训练目标

RL 训练目标为：

```text
J = score_z + CAI_z
```

每一步同义替换的 reward 是：

```text
reward = J(new CDS) - J(old CDS)
```

因此，如果某次替换同时提高模型翻译预测分数和 HEK293T CAI，通常会得到更高奖励。

### 8.2 中心相似性筛选

`CENTER_MIN = 0.70` 仅用于最终候选筛选，不进入训练状态特征，也不参与训练 reward。最终导出的候选需要满足：

```text
s_bio >= 0.70
```

这样可以避免最终序列明显偏离 HEK293T 高表达 CDS 的生物组成分布。

### 8.3 同义约束

优化过程中目标蛋白氨基酸序列固定为 Gluc 序列。每一步只能选择当前位置氨基酸对应的同义密码子进行替换，因此理论上不会改变蛋白质产物。

---

## 9. 常见问题

### 9.1 `ViennaRNA not available`

说明当前环境无法导入 `RNA`。请确认：

```bash
conda activate codonoptimizer
python -c "import RNA; print(RNA.__version__)"
```

如果报错，需要重新安装 ViennaRNA，或在代码中关闭 MFE 相关计算。

### 9.2 GPU 显存不足

降低并行 worker 数量：

```bash
export RL_MAX_PARALLEL_START_JOBS=1
```

或者只指定一张 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
```

### 9.3 结果和旧版本不一致

这是正常的。本版本 RL 内部打分逻辑已经改为：

```text
env_vec = log2(RPKM+1)
scalar mRNA input = median(log2(RPKM+1))
```

如果旧版本使用 `np.log1p(RPKM*5)` 和 `mRNA_count=4.5`，则 reward、score_z、最终候选排序都可能不同。新旧结果不建议混合比较。

### 9.4 输出目录已有旧结果

建议将 `step2_RL_final.py` 中的：

```python
OUT_DIR = "/data/hyliu/HEK293T_RL/RL_results_Gluc"
```

改成新的目录，例如：

```python
OUT_DIR = "/data/hyliu/HEK293T_RL/RL_results_Gluc_scorepy_aligned"
```

这样可以避免旧结果和新打分逻辑下的结果混在一起。

---

## 10. 建议的完整运行流程

```bash
# 1. 激活环境
conda activate codonoptimizer

# 2. 生成随机同义 CDS 起始池
cd /data/hyliu/HEK293T_RL
python step1_generate_gluc_random_cds.py \
  --use_default_gluc \
  --cai_weight_file /data/hyliu/HEK293T_RL/metrics/hek293t_codon_weights.txt \
  --out_dir /data/hyliu/HEK293T_RL/challenge_start_pool \
  --prefix Gluc_MF882921_uniformRandom_seed2026 \
  --n 10000000 \
  --top_k 10 \
  --seed 2026 \
  --max_attempts 20000000 \
  --rare_threshold 0.30 \
  --stop_mode balanced_three \
  --min_codon_diff 8 \
  --calc_mfe_for_selected

# 3. 复制为 step2 默认读取的文件名
cp /data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026.lowestCAI_top10_sequences.txt \
   /data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026_top10_sequences

# 4. 先单起点测试
export CUDA_VISIBLE_DEVICES=0
export RL_SINGLE_START_ID=1
python step2_RL_final.py
unset RL_SINGLE_START_ID

# 5. 正式并行运行
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RL_MAX_PARALLEL_START_JOBS=4
nohup python step2_RL_final.py > step2_RL_final.log 2>&1 &
```

---

