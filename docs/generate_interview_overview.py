from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


OUTPUT = Path(__file__).with_name("VLM-RAG项目面试概述-最终版.docx")


def run(text: str, *, bold: bool = False, size: int = 22, color: str = "000000") -> str:
    props = [
        '<w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="微软雅黑" w:hAnsi="Microsoft YaHei"/>',
        f'<w:color w:val="{color}"/>',
        f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>',
    ]
    if bold:
        props.append("<w:b/><w:bCs/>")
    return f'<w:r><w:rPr>{"".join(props)}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def paragraph(
    text: str,
    *,
    bold: bool = False,
    size: int = 22,
    color: str = "000000",
    align: str | None = None,
    before: int = 0,
    after: int = 55,
    line: int = 260,
    page_break_before: bool = False,
) -> str:
    ppr = [f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>']
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    if page_break_before:
        ppr.append('<w:pageBreakBefore/>')
    return f'<w:p><w:pPr>{"".join(ppr)}</w:pPr>{run(text, bold=bold, size=size, color=color)}</w:p>'


def title(text: str, *, new_page: bool = False) -> str:
    return paragraph(text, bold=True, size=35, color="17365D", align="center", after=70, page_break_before=new_page)


def subtitle(text: str) -> str:
    return paragraph(text, bold=True, size=22, color="666666", align="center", after=110)


def heading(text: str) -> str:
    return paragraph(text, bold=True, size=28, color="1F4E78", before=55, after=45)


def subheading(text: str) -> str:
    return paragraph(text, bold=True, size=23, color="2F5597", before=35, after=30)


def bullet(text: str) -> str:
    return paragraph("• " + text, size=21, after=25, line=250)


def flow(text: str) -> str:
    return paragraph(text, bold=True, size=22, color="17365D", align="center", before=30, after=45, line=255)


body: list[str] = []

# 第 1 页：项目概述与一段式数据流
body += [
    title("VLM-RAG 企业文档智能问答项目"),
    subtitle("第 1 页｜项目概述与完整数据流"),
    heading("1. 项目概述"),
    paragraph("本项目属于人工智能中的企业文档问答与多模态 RAG 方向，目标是处理合同、报表、PPT、单据、手册和制度等图文文档。系统不会把所有文档直接交给大模型，而是先从大量页面中检索出最可能包含答案的证据页，再结合这些页面回答问题，从而降低推理成本，并让答案具有可追溯的页面来源。"),
    heading("2. 核心数据"),
    bullet("Page：一页文档的数据对象，包含 page_id、文档类型、标题、图片路径、页面内容、版式描述和关键事实。"),
    bullet("Query：用户提出的问题；Answer 是标准答案；positive_page_ids 标记真正包含答案的证据页。"),
    bullet("Page、Query、Answer、Evidence Page 共同组成检索训练和问答评测需要的核心数据。"),
    heading("3. 整体数据流"),
    paragraph("系统首先将企业 PDF、PPT 或扫描件按页转换为页面图片，并为每页建立包含页面编号、标题和路径等信息的 Page 数据；随后，页面编码器使用 VLM 或视觉 Embedding 模型提取页面特征，将每页转换成向量并预先写入向量索引。当用户输入 Query 时，文本编码器把问题转换成同一向量空间中的 Query 向量，检索器计算它与页面向量之间的相似度，按照得分选出最相关的 Top-K 页面。候选页面可以继续经过重排器精排，再由 OCR、VLM 或多模态大模型读取其中的文字、表格和图表，结合多个证据页形成最终答案。系统最终返回答案、证据页编号和置信度，并通过标准答案与正确证据页计算 Recall、MRR、EM 和 Accuracy 等指标。"),
    heading("4. 当前原型实际做法"),
    paragraph("当前仓库是一个无需联网和 GPU 的流程原型：data.py 生成 18 个模拟 SVG 页面和 24 条问答，页面真实内容预先保存在 pages.json 的 visual_text 与 facts 字段中；HashingVLMEncoder 用分词和哈希投影代替真实 VLM，WeightedVisualGenerator 用字段匹配代替大模型生成，向量索引也以 JSON 文件保存。因此它证明的是数据构建、编码、检索、回答和评测的工程链路能够运行，不代表已经具备读取真实 PDF 的视觉能力。"),
    heading("5. 项目主要输出"),
    bullet("indexes/page_vectors.json：页面向量；models/retriever_config.json：选出的检索配置。"),
    bullet("outputs/retrieval_results.json：每条 Query 的答案、Top-K 页面、证据页和置信度。"),
    bullet("outputs/metrics_report.csv：VLM-RAG 与 OCR-RAG、SigLIP、ColPali 模拟基线的指标对比。"),
]

# 第 2 页：更详细的核心代码职责
body += [
    title("核心代码逐个说明", new_page=True),
    subtitle("第 2 页｜每个模块的输入、处理和输出"),
    subheading("data.py｜数据模型与模拟数据"),
    bullet("输入：输出目录；处理：定义 Page、Query，生成六类 18 页模拟文档和 24 条问答；输出：pages.json、queries.json、sample_pages/*.svg。"),
    subheading("dataset_split.py｜数据集划分"),
    bullet("输入：Query 列表；处理：按 train/dev/test 比例拆分；输出：splits 下的三个 JSON 和 split_summary.json。"),
    subheading("encoders.py｜Query 与页面编码"),
    bullet("HashingVLMEncoder.encode_query() 对问题分词、中文 bigram、哈希投影和归一化；encode_page() 拼接页面类型、标题、预设文本、版式和 facts 后做同样处理。cosine_similarity() 比较向量；info_nce_loss() 计算对比学习损失。"),
    subheading("retriever.py｜双塔召回"),
    bullet("DualTowerRetriever.index() 预先编码并缓存所有 Page；search() 编码 Query、逐页计算相似度、排序并包装为 SearchHit。它负责“找页”，不负责回答。"),
    subheading("index_store.py｜索引持久化"),
    bullet("build_vector_index() 建页面向量；save/load_vector_index() 把向量和元数据写入或读出 JSON。生产环境这里可替换为 FAISS、Milvus 或 Elasticsearch。"),
    subheading("generator.py｜答案生成占位模块"),
    bullet("WeightedVisualGenerator.answer() 根据 Query 推断所问字段，再结合 Top-K 检索分数、词面重叠和 Page.facts 选择答案，返回 Answer(text、evidence_page_ids、confidence)。它不是 GPT/VLM 生成。"),
    subheading("pipeline.py｜端到端编排"),
    bullet("run_demo() 依次准备数据、建立检索器、逐 Query 检索、生成答案、记录延迟、计算指标，最后输出 retrieval_results.json。"),
    subheading("training.py｜模拟训练与选参"),
    bullet("遍历隐藏层权重组合，用训练 Query 计算 InfoNCE，并在开发集比较 MRR/Recall，保存 retriever_config.json 和 training_log.csv。没有反向传播，也没有真实模型参数更新。"),
    subheading("baselines.py｜对照实验"),
    bullet("ocr_rag 故意制造字符混淆和字段丢失；siglip 只保留标题模拟全局图文向量；colpali 增加 layout token 模拟版式感知。三者都是模拟，不是实际加载对应模型。"),
    subheading("metrics.py｜评测"),
    bullet("recall_at_k() 看正确证据页是否进入前 K；mrr_at_k() 看正确页排名；exact_match()/accuracy() 比较预测答案与标准答案。"),
    subheading("workflows.py、cli.py、config.py、logging_utils.py｜工程组织"),
    bullet("workflows.py 封装各阶段；cli.py 解析 build-data/build-index/train/evaluate/demo/all 命令；config.py 读取 config.yaml；logging_utils.py 写 pipeline.log。"),
]

# 第 3 页：概念、脚本和生产替换关系
body += [
    title("概念与执行链路", new_page=True),
    subtitle("第 3 页｜OCR、VLM、双塔索引、脚本与生产升级"),
    heading("1. 关键概念放到数据流里理解"),
    bullet("OCR：把扫描图片中的像素识别成文字。文字型 PDF 可直接抽字；扫描 PDF 才必须 OCR。OCR 擅长读字，但可能丢失表格行列、图表、颜色、签章和版式。项目中的 ocr_rag 只是错误模拟。"),
    bullet("VLM：视觉语言模型，同时理解页面图片和自然语言。真实页面编码时可保留文字、表格、图表和布局。当前 HashingVLMEncoder 是接口占位，不是真实 VLM。"),
    bullet("Query：用户问题。训练数据中还要有 Answer 和 Evidence Page，形成“问题—答案—证据页”监督。"),
    bullet("双塔：文本塔把 Query 编成 q，视觉塔把页面编成 p，两者映射到同一向量空间，用 sim(q,p) 排序。它不是“双层索引”。"),
    bullet("向量索引：离线保存所有 p；在线只计算 q，再找最相似页面。双塔速度快、适合召回；重排器对少量候选做二次精排，通常更准但更慢。"),
    bullet("Top-K：相似度最高的 K 个页面。本项目 K=3；过小可能漏证据，过大会增加噪声和 VLM 推理成本。"),
    heading("2. 脚本执行顺序及产物"),
    flow("build_dataset → build_index → train_retriever → evaluate → run_demo"),
    bullet("build_dataset.py：生成 data/pages.json、queries.json、SVG 页面及训练/验证/测试划分。"),
    bullet("build_index.py：读取页面并生成 indexes/page_vectors.json、index_metadata.json。"),
    bullet("train_retriever.py：搜索权重并生成 models/retriever_config.json、training_log.csv。"),
    bullet("evaluate.py：运行四种模拟方案，生成 outputs/metrics_report.csv。"),
    bullet("run_demo.py：执行端到端问答，生成 outputs/retrieval_results.json；vlm_rag_cli.py all 可一次执行全部步骤。"),
    heading("3. 当前实现与真实落地的替换关系"),
    bullet("模拟 SVG → 真实 PDF/PPT 页面渲染；pages.json 预设文字 → OCR/VLM 实际读取页面。"),
    bullet("HashingVLMEncoder → ColPali/ColQwen、SigLIP 或其他文档视觉 embedding；JSON 索引 → FAISS/Milvus。"),
    bullet("WeightedVisualGenerator → Qwen-VL、InternVL、MiniCPM-V 或云端多模态模型；模拟问答 → 企业脱敏问答对。"),
    heading("4. 面试总结（可直接说）"),
    paragraph("“这是一个企业文档 VLM-RAG 原型。我把系统拆成离线建库和在线问答两条数据流：离线把页面编码并建立向量索引，在线把 Query 编码后用双塔召回 Top-K 证据页，再生成答案并返回引用。仓库目前用哈希向量和 facts 规则把全链路跑通，OCR、VLM、SigLIP、ColPali 都是模拟或预留接口。真实落地时需要接入 PDF 渲染、真实视觉模型、向量数据库，并用 Query—答案—证据页数据微调检索器或重排器。”"),
]


document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="800" w:right="900" w:bottom="800" w:left="900" w:header="420" w:footer="420" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>'''

styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="微软雅黑" w:hAnsi="Microsoft YaHei"/><w:sz w:val="22"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="55" w:line="260" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
</w:styles>'''

content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

document_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>'''

settings_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:zoom w:percent="100"/><w:defaultTabStop w:val="420"/><w:characterSpacingControl w:val="doNotCompress"/>
</w:settings>'''

now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
core_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>VLM-RAG 项目面试概述</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''

app_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application><AppVersion>16.0000</AppVersion><Pages>3</Pages><Company></Company>
</Properties>'''

with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
    archive.writestr("[Content_Types].xml", content_types)
    archive.writestr("_rels/.rels", root_rels)
    archive.writestr("word/document.xml", document_xml)
    archive.writestr("word/styles.xml", styles_xml)
    archive.writestr("word/settings.xml", settings_xml)
    archive.writestr("word/_rels/document.xml.rels", document_rels)
    archive.writestr("docProps/core.xml", core_xml)
    archive.writestr("docProps/app.xml", app_xml)

print(OUTPUT)
