# BRCA 恶性细胞转移隐空间模型

该目录包含全量 120,063 个严格恶性细胞的训练、整样本留出评价、样本级预测图、基因归因和稳定性审计代码。

## 1. 数据要求

| 文件 | 用途 |
|---|---|
| `BRCA_all_data.h5ad` | 全部 BRCA 单细胞计数 |
| `BRCA_all_fastCNV_Epithelial.h5ad` | `cnv_fraction` 及 CNV 审计 |
| `cell_metadata.csv.gz` | 严格恶性细胞的元数据和原 h5ad 行号 |
| `balanced_malignant_counts_4k.h5ad` | 固定基因集及原 44K 模型的公平留出划分 |

恶性细胞定义为：Epithelial、CNV-high、EPCAM/KRT8/KRT18/KRT19 至少一个表达、PTPRC=0。

## 2. 创建环境

```bash
conda env create -f environment.yml
conda activate brca-metastasis-atlas
cp config.env.example config.env
```

如果在原服务器目录上运行，`config.env.example` 中已填写当前路径。GPU 编号可通过 `BRCA_DEVICE=cuda:0` 调整。

## 3. 运行方式

### 只复现已训练模型的留出预测图

```bash
bash run_pipeline.sh evaluate
```

输出：

- `full_balanced_split_v4b/heldout_sample_predictions.csv`
- `full_balanced_split_v4b/heldout_cell_predictions.csv.gz`
- `full_balanced_split_v4b/heldout_sample_summary.json`
- `full_balanced_split_v4b/heldout_sample_predictions.png`

### 从头训练全量模型

```bash
bash run_pipeline.sh train
```

训练使用与 44K 模型相同的整 GSM 留出样本。同一 GSM 的所有细胞只能出现在训练集或验证集的一侧。

### 重新执行基因归因和证据审计

```bash
bash run_pipeline.sh genes
```

### 执行全部流程

```bash
bash run_pipeline.sh all 2>&1 | tee run_all.log
```

## 4. 主要脚本

| 脚本 | 功能 |
|---|---|
| `train_full_malignant_v4.py` | 构建/读取 120K 严格恶性细胞缓存 |
| `train_full_balanced_split_v4b.py` | 训练批次、通用转移、器官特异三隐空间模型 |
| `evaluate_heldout_samples_v4b.py` | 整 GSM 留出预测、样本聚合和条形图 |
| `explain_full_v4b.py` | 积分梯度和虚拟扰动基因归因 |
| `validate_gene_stability_v4b.py` | 样本伪 bulk、跨研究效应、bootstrap 和置换检验 |
| `refine_gene_candidates_v4b.py` | CNV 关联和谱系污染审计 |
| `integrate_gene_evidence_v4b.py` | 整合 contrastiveVI、TCGA 和单细胞证据 |

## 5. 当前复现指标

- 细胞级转移 AUC：0.939
- 细胞级平衡准确率：0.904
- 样本级 AUC：0.813
- 19 个可评价留出样本中 15 个分类正确，准确率 78.9%
- 转移样本器官准确率：68.8%

## 6. 注意

评价单位必须是患者或样本。细胞级高 AUC 不能替代样本级外推。当前通用转移隐空间仍包含器官信息，因此模型适合作为转移状态研究工具，尚不应直接用于临床决策。

