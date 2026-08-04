# VLM-RAG 页面图像检索增强生成演示项目

这是一个可直接运行的企业内部多模态大模型算法原型工程，主题是“基于页面图像的视觉检索增强生成（VLM-RAG）”。

本项目为了便于本地快速跑通，不依赖在线大模型或 GPU。代码用标准库实现了一个轻量版流程：

- 自动生成六类企业文档页面图像样例，覆盖合同、报表、PPT、单据、手册、制度
- 使用文本 Query 与页面图像统一编码
- 基于 InfoNCE 风格目标实现双塔检索器训练/权重搜索
- Top-K 页面检索、分数加权、多页答案融合
- 输出 MRR@10、Recall@K、EM、Accuracy 和基线对比表

真实生产环境中，可将 `src/vlm_rag/encoders.py` 替换为 MiniCPM-V、SigLIP、ColPali 或 GPT-4o 相关编码/生成接口。

## 快速运行

环境要求：

- Python 3.10+
- 无需联网
- 无需 GPU
- 无需额外安装第三方依赖

```bash
python3 scripts/run_demo.py
```

运行后会生成：

- `data/sample_pages/*.svg`：模拟企业图文页面
- `outputs/retrieval_results.json`：检索与问答结果
- 控制台指标：MRR@10、Recall@3、EM、Accuracy

```bash
python3 scripts/run_demo.py --data-dir data --output-dir outputs
```

## 完整实验流程

```bash
python3 scripts/build_dataset.py
python3 scripts/build_index.py
python3 scripts/train_retriever.py
python3 scripts/evaluate.py
python3 scripts/run_demo.py
```

## 真实数据格式准备

真实模型链路统一使用页面级 JSONL 数据：

- `pages.jsonl`：页面编号、文档编号、文档类型、页码、图片路径和元数据。
- `samples.jsonl`：Query、一个或多个标准答案、一个或多个证据页、数据划分和元数据。
- `manifest.json`：数据规模、文档类型、数据划分和校验警告。

先将当前演示数据转换为统一格式，验证后续真实数据接口：

```bash
python3 scripts/prepare_real_dataset.py
```

默认输出到 `data/canonical/`。数据规范位于 `src/vlm_rag/dataset_schema.py`，来源数据转换逻辑位于 `src/vlm_rag/dataset_adapters.py`。后续 DocVQA、ChartQA 等数据集都转换到同一格式，再进入模型编码与检索流程。

接入导师指定的 ChartQA 小样本：

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-real.txt
.venv\Scripts\python scripts\download_chartqa.py
```

默认从 `HuggingFaceM4/ChartQA` 的 train、val、test 分别读取 60、20、20 条记录，图片按内容哈希去重，并输出到 `data/real/chartqa/`。可以使用 `--train-rows`、`--dev-rows` 和 `--test-rows` 调整规模。

## 真实 SigLIP 特征提取

真实模型链路使用独立的 `.venv-models` 环境。先按照 PyTorch 官网为电脑安装支持 CUDA 的 PyTorch，再安装模型依赖：

```bash
py -3.13 -m venv --system-site-packages .venv-models
.venv-models\Scripts\python -m pip install -r requirements-models.txt
```

运行一条 Query 和两张 ChartQA 页面图片的冒烟测试：

```bash
.venv-models\Scripts\python scripts\smoke_test_siglip.py
```

首次运行会从 Hugging Face 下载 `google/siglip-base-patch16-224`。`src/vlm_rag/siglip_encoder.py` 用模型的文本塔编码 Query、用图像塔编码页面，并将两种 768 维特征归一化到同一向量空间；之后可以直接计算余弦相似度并进行 Top-K 检索。

批量编码所有 ChartQA 页面、保存真实向量索引并评估全部 Query：

```bash
.venv-models\Scripts\python scripts\evaluate_siglip_retrieval.py
```

页面向量保存在 `indexes/siglip_chartqa/`，每条 Query 的 Top-10 排名和汇总指标分别保存在 `outputs/siglip_chartqa/retrieval_results.json` 与 `outputs/siglip_chartqa/metrics.json`。索引存在时会直接复用；模型或页面发生变化时添加 `--rebuild-index` 重建。

当前50张页面、100条Query上的零样本 SigLIP 基线结果如下。这里尚未使用项目问答对微调模型：

| 指标 | 结果 |
| --- | ---: |
| Recall@1 | 0.2100 |
| Recall@3 | 0.3800 |
| Recall@10 | 0.6300 |
| MRR@10 | 0.3356 |

## OCR-RAG 基线

OCR单独使用Python 3.11环境，避免PaddlePaddle与PyTorch模型环境发生依赖冲突：

```bash
py -3.11 -m venv .venv-ocr
.venv-ocr\Scripts\python -m pip install paddlepaddle==3.3.1 --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/
.venv-ocr\Scripts\python -m pip install paddleocr==3.7.0
.venv-ocr\Scripts\python scripts\extract_chartqa_ocr.py
```

脚本使用本地PP-OCR识别页面文字，结果逐页写入 `data/real/chartqa/ocr_text.jsonl`，中断后再次运行会自动跳过已完成页面。

OCR完成后，在模型环境中使用BGE文本嵌入模型建立索引并评估：

```bash
.venv-models\Scripts\python -m pip install sentence-transformers==5.6.1
.venv-models\Scripts\python scripts\evaluate_ocr_retrieval.py
```

结果保存在 `outputs/ocr_bge_chartqa/`，与SigLIP的统一对比表保存在 `outputs/baseline_retrieval_comparison.csv`。

## ColPali系列晚交互基线

原版 `vidore/colpali-v1.3` 是3B模型，本机8GB显存先使用同系列的官方小模型 `vidore/colSmol-500M` 验证多向量晚交互检索：

```bash
.venv-colpali\Scripts\python -m pip install -r requirements-colpali.txt
.venv-colpali\Scripts\python scripts\evaluate_colsmol_retrieval.py
```

该方法为Query和页面保留多组token向量，并使用MaxSim晚交互打分。页面索引保存在 `indexes/colsmol_chartqa/`，指标保存在 `outputs/colsmol_chartqa/`。

当前三种真实检索基线使用相同的50张ChartQA页面和100条Query：

| 方法 | Recall@1 | Recall@3 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| SigLIP零样本图文检索 | 0.21 | 0.38 | 0.63 | 0.3356 |
| PP-OCR + BGE文本检索 | 0.26 | 0.32 | 0.48 | 0.3085 |
| ColSmol多向量晚交互 | 0.34 | 0.53 | 0.68 | 0.4455 |

其中ColSmol页面编码耗时44.4秒，100条Query编码与打分耗时2.1秒，峰值GPU显存约1.11GB。原版3B ColPali留待更大显存服务器上的正式实验。

## 单页视觉问答基线

使用官方 `HuggingFaceTB/SmolVLM-500M-Instruct`，先将标注的正确证据页直接交给模型回答，单独评估生成能力：

```bash
.venv-colpali\Scripts\python scripts\evaluate_oracle_page_qa.py
```

预测逐条保存在 `outputs/oracle_page_qa/predictions.jsonl`，支持中断续跑；汇总输出精确匹配率、ChartQA风格的5%数值容差准确率和平均单题耗时。

使用ColSmol检索的Top-3页面进行图片拼接问答：

```bash
.venv-colpali\Scripts\python scripts\evaluate_topk_collage_qa.py
```

结果保存在 `outputs/top3_collage_qa/`，用于与后续“Top-K逐页推理再融合”的主方案比较。

当前生成基线结果：

| 输入方式 | 检索召回 | EM | 5%容差准确率 | 平均单题耗时 |
| --- | ---: | ---: | ---: | ---: |
| 标注的正确证据页 | 1.00 | 0.35 | 0.41 | 1.42秒 |
| ColSmol Top-3图片拼接 | 0.53 | 0.17 | 0.23 | 0.71秒 |
| ColSmol Top-3逐页推理与加权融合 | 0.53 | 0.19 | 0.26 | 3.46秒 |

正确证据页实验用于隔离生成能力；后两项包含检索误差。逐页推理比图片拼接的EM提高2个百分点、5%容差准确率提高3个百分点，但因为每题需要推理3次，平均耗时增加到3.46秒。

## InfoNCE轻量检索器训练

冻结SigLIP主模型，复用已提取的768维特征，只训练文本和页面两侧的低秩残差投影层：

```bash
.venv-models\Scripts\python scripts\train_siglip_infonce.py
```

训练只使用train问答对，使用dev的MRR@10选择最佳轮次，test仅作最终评估。模型权重和训练日志保存在 `models/siglip_infonce/`，检索结果保存在 `outputs/siglip_infonce/`。

当前60条训练Query上的共享特征权重版本将整体MRR@10从0.3356提升到0.3434、Recall@1从0.21提升到0.22；提升较小，说明需要继续扩充训练问答和hard negatives。

## Top-K逐页问答与融合

将ColSmol召回的Top-3页面分别交给SmolVLM回答，再根据检索分数对相同候选答案加权投票：

```bash
.venv-colpali\Scripts\python scripts\evaluate_topk_sequential_qa.py
```

逐页候选支持断点续跑，保存在 `outputs/top3_sequential_qa/page_candidates.jsonl`；融合答案和指标保存在同一目录。

当前100条Query的完整评测结果为：Top-3检索召回率0.53、EM 0.19、5%数值容差准确率0.26、平均单题生成耗时3.46秒。该结果说明逐页读取可以减少拼接图片造成的缩放和视觉拥挤，但最终效果仍受Top-3检索召回率限制。

当前PP-OCRv6-medium与BGE-small组合的真实基线结果如下：

| 指标 | 结果 |
| --- | ---: |
| Recall@1 | 0.2600 |
| Recall@3 | 0.3200 |
| Recall@10 | 0.4800 |
| MRR@10 | 0.3085 |

50张图表的OCR平均单页耗时约25156毫秒。该耗时来自本机Windows CPU顺序推理，仅作为当前实验环境下的工程参考。

## 轻量多向量视觉检索基线

本机8GB显存先使用官方 `vidore/colSmol-500M` 验证ColBERT式late interaction多向量检索。该结果必须标记为ColSmol，不能当作原版3B ColPali成绩：

```bash
py -3.13 -m venv --system-site-packages .venv-colvision
.venv-colvision\Scripts\python -m pip install -r requirements-colvision.txt
.venv-colvision\Scripts\python scripts\evaluate_colsmol_retrieval.py
```

页面多向量索引保存在 `indexes/colsmol_chartqa/`，指标保存在 `outputs/colsmol_chartqa/`。原版 `vidore/colpali-v1.3` 基础权重约5.87GB，计划在显存更充足的服务器上进行正式基线复现。

当前落盘并可复现的ColSmol轻量多向量检索结果：Recall@1为0.3400、Recall@3为0.5300、Recall@10为0.6800、MRR@10为0.4455。50页编码耗时44.4秒，100条Query编码与打分耗时2.1秒，峰值显存约1.11GB。

使用SmolVLM分别评估正确证据页、检索Top-1页和Top-3拼接图问答：

```bash
.venv-colvision\Scripts\python scripts\evaluate_smolvlm_qa.py
```

预测逐条保存在 `outputs/smolvlm_qa/predictions.jsonl`，任务中断后可继续运行。汇总同时报告严格EM和ChartQA数值答案允许5%误差的 relaxed accuracy。

当前100条Query的生成基线结果：

| 输入方式 | EM | Relaxed Accuracy |
| --- | ---: | ---: |
| 正确证据页（Oracle） | 0.4700 | 0.5400 |
| ColSmol检索Top-1页 | 0.2100 | 0.3200 |
| ColSmol检索Top-3拼接图 | 0.1800 | 0.2900 |

Oracle结果用于观察生成模型自身的读图上限；检索Top-1结果反映当前端到端效果；拼接结果用于验证简单图片拼接是否有效。

## InfoNCE轻量检索适配

冻结SigLIP主干，只训练文本塔与图像塔后的低秩残差投影层：

```bash
.venv-models\Scripts\python scripts\train_siglip_adapter.py
```

脚本只使用train问答计算InfoNCE，默认冻结页面塔并仅训练Query侧低秩适配层，使用dev MRR@10进行早停与最佳轮次选择，最后单独报告test指标。模型保存在 `outputs/siglip_adapter_query/adapter.pt`，训练日志与指标保存在同一目录。添加 `--train-towers both` 可以实验双侧适配。

中等规模实验使用600/100/100条train/dev/test问答和386张无跨划分泄漏页面。共享适配层的test结果为：Recall@1从0.23提升到0.27、Recall@3从0.28提升到0.32、Recall@10从0.37提升到0.41、MRR@10从0.2661提升到0.3060。结果位于 `outputs/siglip_adapter_medium_shared/`。

阶段三前半段已使用固定随机种子重新复现上述结果，当前最佳权重位于 `outputs/stage3_local_best/adapter.pt`，详细实验设置与消融结论见 `docs/stage3_local_training_report.md`。

Top-15 hard-negative消融实验的test MRR@10为0.2937、Recall@1为0.25，虽然高于零样本基线，但低于全部训练页面作为负例的方案，因此不作为当前最佳检查点。

## Top-3逐页问答与融合

对检索Top-3页面分别运行视觉问答，记录每页候选答案、检索分数和生成平均token置信度，再进行加权投票：

```bash
.venv-colvision\Scripts\python scripts\evaluate_smolvlm_multipage.py
```

逐页候选支持断点续跑，保存在 `outputs/smolvlm_multipage/page_candidates.jsonl`；融合结果与指标保存在同一目录。

当前100条Query的结果为：证据页Recall@3为0.53，融合答案EM为0.21，5%数值容差准确率为0.32，候选答案Oracle准确率为0.42。融合结果暂未超过直接使用Top-1页面（0.32），说明下一步应训练或引入候选答案重排器，而不是继续在测试集上调整固定权重。

## 本地检索API与Dify接入点

阶段五已先完成可调用的检索API。它把自然语言Query编码为SigLIP向量，在已经落盘的页面索引中检索，并返回Top-K证据页面、相似度分数和图片路径。安装并启动：

```bash
.venv-models\Scripts\python -m pip install -r requirements-api.txt
.venv-models\Scripts\python scripts\run_api.py
```

服务默认监听 `http://127.0.0.1:8000`，接口包括：

- `GET /health`：检查索引、模型名称和模型加载状态。
- `POST /v1/search`：请求体为 `{"query":"用户问题","top_k":3}`，返回证据页面列表。
- `GET /docs`：FastAPI自动生成的接口调试页面。

Dify后续可在工作流中添加HTTP请求节点调用 `/v1/search`。目前只完成本地API封装，尚未在Dify平台创建工作流，也没有把本地图片公开到外网。

各脚本作用：

- `scripts/build_dataset.py`：生成 18 页模拟企业图文页面、24 条问答样本和 train/dev/test 划分。
- `scripts/build_index.py`：构建并落盘页面向量索引，输出 `indexes/page_vectors.json` 和 `indexes/index_metadata.json`。
- `scripts/train_retriever.py`：用 InfoNCE 目标对隐藏层池化权重做轻量训练/搜索，输出 `models/retriever_config.json` 和 `models/training_log.csv`。
- `scripts/evaluate.py`：评估 VLM-RAG、OCR-RAG、SigLIP、ColPali 四种方案，输出 `outputs/metrics_report.csv`。
- `scripts/run_demo.py`：跑端到端检索增强生成，输出 `outputs/retrieval_results.json`。

也可以使用统一 CLI：

```bash
python3 scripts/vlm_rag_cli.py all
```

## 当前模拟实验结果

`scripts/evaluate.py` 会生成如下对比维度：

| method    | 含义                                   |
| --------- | -------------------------------------- |
| `vlm_rag` | 页面图像 VLM-RAG 主方案                |
| `ocr_rag` | OCR 文本链路模拟基线                   |
| `siglip`  | 全局图文向量检索模拟基线               |
| `colpali` | 版式感知 late-interaction 检索模拟基线 |

本项目中的基线是离线可运行模拟，用来展示实验框架和指标计算。真实项目中可以在 `src/vlm_rag/baselines.py` 替换为实际模型调用。

## 项目结构

```text
.
├── README.md
├── configs
│   └── config.yaml
├── docs
│   └── technical_report.md
├── scripts
│   ├── build_index.py
│   ├── build_dataset.py
│   ├── evaluate.py
│   ├── run_demo.py
│   ├── train_retriever.py
│   └── vlm_rag_cli.py
└── src
    └── vlm_rag
        ├── baselines.py
        ├── cli.py
        ├── config.py
        ├── dataset_split.py
        ├── __init__.py
        ├── data.py
        ├── encoders.py
        ├── generator.py
        ├── index_store.py
        ├── logging_utils.py
        ├── metrics.py
        ├── pipeline.py
        ├── retriever.py
        ├── training.py
        └── workflows.py
```

## 文档

- 技术方案文档：`docs/technical_report.md`
