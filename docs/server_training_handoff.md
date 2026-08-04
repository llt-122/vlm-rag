# 服务器训练交接说明

## 上传前状态

当前训练包使用 `data/real/chartqa_medium`，包含 386 张页面图像和 800 条问答，划分为 600 条训练、100 条验证、100 条测试。数据采用统一结构：

- `pages.jsonl`：页面编号、文档编号、文档类型、页码、图像路径和元数据。
- `samples.jsonl`：Query、答案、证据页、文档类型和 train/dev/test 划分。

正式企业数据也必须转换成相同结构，并通过严格校验后才能训练。

## 建议上传内容

上传项目代码以及以下训练数据：

```text
configs/server_training.json
data/real/chartqa_medium/
docs/server_training_handoff.md
requirements-models.txt
requirements-real.txt
requirements-server.txt
scripts/
src/
tests/
```

不需要上传 `.venv*`、`.cache`、`outputs`、`features` 和 `indexes`。它们要么与本机环境绑定，要么可以在服务器重新生成。

上传前运行以下命令生成逐文件 SHA-256 校验清单：

```bash
python scripts/create_server_upload_manifest.py
```

生成的 `server_upload_manifest.json` 也应一起上传，用于核对文件是否完整。

## 服务器执行顺序

建议使用带 CUDA 的 Linux GPU 服务器。先按服务器 CUDA 版本安装 PyTorch，再执行：

```bash
python -m venv .venv-server
source .venv-server/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-server.txt
python -m pytest -q
python scripts/run_training_pipeline.py --config configs/server_training.json --dry-run
bash scripts/server_train.sh
```

流水线会依次完成数据严格校验、SigLIP 页面特征提取与零样本评测、Query 特征提取、InfoNCE 轻量适配器训练、dev 早停和 test 最终评测。训练结果写入 `outputs/server_training/`。

## 换成企业数据

企业文档应先脱敏、按页转成图像并标注 Query、答案和证据页。必须按 `doc_id` 划分数据，禁止同一文档同时出现在 train、dev、test。完成转换后修改 `configs/server_training.json` 的 `dataset_dir`，运行：

```bash
python scripts/validate_training_dataset.py --dataset-dir data/real/enterprise
```

只有报告中的 `status` 为 `ready` 才开始正式训练。

## 训练完成后需要带回的文件

```text
outputs/server_training/adapter.pt
outputs/server_training/metrics.json
outputs/server_training/training_log.csv
outputs/server_training/dataset_validation.json
outputs/server_training/run_manifest.json
```

其中 `adapter.pt` 是权重，`metrics.json` 和 `training_log.csv` 用于比较训练前后的 MRR@10、Recall@K 与训练过程。
