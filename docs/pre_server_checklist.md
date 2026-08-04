# 服务器训练前检查结果

截至 2026-08-04，公开数据训练包已经完成以下检查：

- 已统一页面图像、Query、答案、证据页数据结构。
- ChartQA medium 包含 386 张页面、800 条问答（600 train / 100 dev / 100 test）。
- train、dev、test 按页面/文档隔离，没有文档泄漏。
- 页面图像均存在且不是空文件。
- SigLIP 页面特征提取和零样本检索可以在 CUDA 上运行。
- InfoNCE 轻量适配器可以训练、早停并保存检查点。
- 本地端到端冒烟测试已产出 `adapter.pt`、训练日志、指标和运行清单。
- 服务器训练配置和 Linux 一键启动脚本已经准备完成。
- 上传文件 SHA-256 清单可以通过 `scripts/create_server_upload_manifest.py` 重新生成。

当前边界：训练包使用公开 ChartQA 数据验证技术流程，并不等同于企业数据。拿到脱敏企业文档后，仍需按照 `docs/enterprise_data_format.md` 标注并通过同一严格校验；代码和训练流程不需要重写。
