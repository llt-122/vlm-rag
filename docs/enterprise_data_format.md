# 企业文档训练数据格式

正式训练前，企业文档必须先脱敏并按页渲染为 PNG/JPEG。页面图像是检索单元，不能只保留 OCR 文本。

`pages.jsonl` 每行表示一页：

```json
{"page_id":"contract_001_p005","doc_id":"contract_001","doc_type":"contract","page_no":5,"image_path":"data/real/enterprise/pages/contract_001_p005.png","title":"采购合同第5页","metadata":{"source":"desensitized_enterprise"}}
```

`samples.jsonl` 每行表示一条问答：

```json
{"query_id":"contract_q_000001","query":"合同约定的付款期限是多少？","answers":["收到发票后30日内"],"evidence_page_ids":["contract_001_p005"],"doc_type":"contract","split":"train","metadata":{"source":"human_annotation"}}
```

标注要求：

- `evidence_page_ids` 必须确实包含答案，不能只填写文档编号。
- 同一文档的全部页面和问答只能属于一个 split。
- Query 应覆盖事实、定位、比较和跨页问题；模糊或无法从证据页回答的问题应删除。
- 跨页问题可填写多个 `evidence_page_ids`。
- 答案存在多种合理写法时，可在 `answers` 中填写多个答案。
- 企业数据应先脱敏，不要把姓名、身份证号、账号等敏感信息上传到外部服务器。

转换完成后运行严格校验：

```bash
python scripts/validate_training_dataset.py --dataset-dir data/real/enterprise
```

校验报告为 `ready` 后，才能把配置中的 `dataset_dir` 改为企业数据目录并开始训练。
