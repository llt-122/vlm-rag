from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


OUTPUT = Path(__file__).with_name("VLM-RAG项目说明-论文版-文字说明.docx")


def make_run(text: str, *, bold: bool = False, size: int = 21, font: str = "宋体") -> str:
    props = [
        f'<w:rFonts w:ascii="Times New Roman" w:eastAsia="{font}" w:hAnsi="Times New Roman"/>',
        f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>',
        '<w:color w:val="000000"/>',
    ]
    if bold:
        props.append('<w:b/><w:bCs/>')
    return f'<w:r><w:rPr>{"".join(props)}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def paragraph(
    text: str,
    *,
    size: int = 21,
    bold: bool = False,
    align: str = "both",
    first_line: bool = True,
    before: int = 0,
    after: int = 20,
    line: int = 245,
    font: str = "宋体",
    page_break_before: bool = False,
) -> str:
    props = [
        f'<w:jc w:val="{align}"/>',
        f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>',
    ]
    if first_line:
        props.append('<w:ind w:firstLine="480"/>')
    if page_break_before:
        props.append('<w:pageBreakBefore/>')
    return f'<w:p><w:pPr>{"".join(props)}</w:pPr>{make_run(text, bold=bold, size=size, font=font)}</w:p>'


def mixed_paragraph(label: str, text: str, *, first_line: bool = False) -> str:
    props = [
        '<w:jc w:val="both"/>',
        '<w:spacing w:before="0" w:after="25" w:line="245" w:lineRule="auto"/>',
    ]
    if first_line:
        props.append('<w:ind w:firstLine="480"/>')
    return (
        f'<w:p><w:pPr>{"".join(props)}</w:pPr>'
        f'{make_run(label, bold=True, size=21, font="黑体")}'
        f'{make_run(text, size=21, font="宋体")}</w:p>'
    )


def section(text: str, *, new_page: bool = False) -> str:
    return paragraph(
        text,
        size=26,
        bold=True,
        align="left",
        first_line=False,
        before=45,
        after=30,
        line=260,
        font="黑体",
        page_break_before=new_page,
    )


def subsection(text: str) -> str:
    return paragraph(
        text,
        size=22,
        bold=True,
        align="left",
        first_line=False,
        before=30,
        after=20,
        line=250,
        font="黑体",
    )


body: list[str] = [
    paragraph(
        "基于页面图像的企业文档 VLM-RAG 项目说明",
        size=40,
        bold=True,
        align="center",
        first_line=False,
        before=40,
        after=110,
        line=360,
        font="黑体",
    ),
    mixed_paragraph(
        "摘要：",
        "本项目是面向企业合同、报表、PPT、单据、手册和制度等图文文档的视觉检索增强生成原型。系统以文档页面为基本检索单元，通过统一编码用户问题与页面内容，召回相关证据页并生成答案，同时输出证据页和评测指标。当前仓库使用模拟页面、哈希向量和规则答案实现端到端流程，用于验证数据构建、页面索引、双塔检索、答案生成与评测之间的工程关系。",
    ),
    mixed_paragraph(
        "关键词：",
        "VLM-RAG；企业文档问答；OCR；双塔检索；向量索引；Query",
    ),
    section("1  项目概述"),
    paragraph("RAG（Retrieval-Augmented Generation，检索增强生成）的基本思想是先从外部知识库中检索与问题相关的内容，再依据检索结果生成答案。该项目进一步将检索对象从普通文本片段扩展为完整文档页面，使表格、图表、版式和签章等视觉信息能够被保留。项目输入是企业文档和用户问题，主要输出是问题答案、证据页面、检索排名与置信度。"),
    paragraph("仓库根目录由数据、代码、脚本和运行产物组成。data 保存页面、Query 和数据集划分；src/vlm_rag 保存核心算法；scripts 提供各阶段执行入口；indexes 保存页面向量；models 保存检索配置与训练日志；outputs 保存检索结果与指标报告；configs 和 logs 分别保存参数与运行日志。"),
    subsection("1.1  当前原型的实现范围"),
    paragraph("当前项目没有直接读取真实 PDF，也未实际调用 OCR、GPT-4o、SigLIP、ColPali 或其他 VLM。data.py 生成 18 个 SVG 模拟页面和 24 条问答，页面文字与答案事实预先存放在 pages.json 中；HashingVLMEncoder 使用分词、哈希投影和向量归一化代替真实模型；WeightedVisualGenerator 根据问题和 facts 字段选择答案。因此当前指标只能说明小型模拟流程已经跑通，不能代表真实文档理解效果。"),
    section("2  核心概念"),
    subsection("2.1  Query、Page、Answer 与 Evidence Page"),
    paragraph("Query 表示用户问题，例如“采购合同首付款比例是多少”；Page 表示单个文档页面，包含 page_id、标题、页面路径、文本描述、版式和关键事实；Answer 表示标准答案或预测答案；Evidence Page 表示真正包含答案的证据页。Query 与 Evidence Page 的对应关系是训练检索器最重要的监督信号。"),
    subsection("2.2  OCR、VLM 与 Embedding"),
    paragraph("OCR 用于把扫描页中的像素转换为文字，适合处理无法直接提取文本的 PDF，但可能破坏表格行列、图表和版式关系。VLM 是能够同时理解图像和语言的视觉语言模型，可直接接收页面图片与问题。Embedding 是模型生成的语义向量，用于表示 Query 或页面内容；它不同于图片文件的二进制编码，向量距离反映的是语义相关程度。"),
    subsection("2.3  双塔检索、向量索引与重排"),
    paragraph("双塔检索由文本塔和页面塔组成：文本塔把 Query 编成向量 q，页面塔把 Page 编成向量 p，两者映射到同一向量空间，再通过余弦相似度进行排序。页面向量可以提前计算并写入向量索引，因此在线查询速度较快。Top-K 表示首先召回得分最高的 K 个页面；重排器则对少量候选页进行更精细的二次排序，以提高正确证据页的排名。"),
    subsection("2.4  InfoNCE 与评测指标"),
    paragraph("InfoNCE 是双塔检索常用的对比学习目标，其作用是拉近 Query 与正例证据页的向量，推远 Query 与负例页面的向量。Recall@K 判断正确证据页是否进入前 K 名；MRR 衡量正确页面的排序位置；EM 判断预测答案与标准答案是否完全一致；Accuracy 表示整体回答正确率。"),

    section("3  系统整体数据流", new_page=True),
    paragraph("系统首先将企业 PDF、PPT 或扫描件按页转换成页面图片，并为每页建立包含页面编号、文档类型、标题、图片路径和页面内容的 Page 数据。随后，页面编码器使用 VLM 或视觉 Embedding 模型提取页面特征，将页面转换为向量并写入向量索引。当用户输入 Query 后，文本编码器将问题转换为同一向量空间中的 Query 向量，检索器计算该向量与页面向量之间的余弦相似度，按得分召回最相关的 Top-K 页面。候选页面可继续通过重排器调整顺序，再由 OCR、VLM 或多模态生成模型读取证据页中的文字、表格和图表，并结合多个页面形成最终答案。系统最终返回答案、证据页编号、检索分数和置信度，并依据标准答案及正确证据页计算检索与问答指标。在当前原型中，真实页面编码由哈希向量代替，页面理解和答案生成由 pages.json 中的预设文字与 facts 字段代替，但各阶段的数据传递关系保持一致。"),
    section("4  核心代码分析"),
    mixed_paragraph("data.py：", "项目开始运行时，首先要准备能够被程序统一处理的页面和问题。这个文件把一页合同、报表或手册整理成带编号、标题、图片路径、页面文字和关键事实的 Page，同时把用户问题、标准答案和正确证据页整理成 Query。当前项目没有接入真实企业文档，因此它直接生成 18 个模拟页面和 24 条问答，并把这些内容保存到 pages.json、queries.json 和 SVG 文件中，供后续编码与检索使用。"),
    mixed_paragraph("dataset_split.py：", "问答数据生成以后，不能全部拿来训练或全部拿来测试。这个文件按照配置比例把问题分成训练集、验证集和测试集：训练集用于选择检索参数，验证集用于比较哪组参数效果更好，测试集用于最后评估。划分结果保存在 data/splits 中，后面的训练和评测流程会直接读取这些文件。"),
    mixed_paragraph("encoders.py：", "页面和问题准备好以后，需要把文字内容转换成可以比较的数字向量。这个文件先对 Query 和页面内容分词，再用哈希方式把词语投影到 384 维空间，并对向量进行归一化。这样，“付款比例是多少”与包含付款信息的页面会产生一定的共同特征。这里的编码过程只是为了模拟真实 VLM 或 Embedding 模型的接口，并没有真正查看 SVG 图片。"),
    mixed_paragraph("retriever.py：", "得到页面向量后，这个文件负责真正执行“找页面”。建立索引时，它提前保存每个页面的向量；收到 Query 时，它先生成问题向量，再与所有页面向量计算余弦相似度，把得分从高到低排列，最后返回最相关的 Top-K 页面以及每页的分数和名次。它解决的是答案可能在哪一页的问题，并不直接回答用户。"),
    mixed_paragraph("index_store.py：", "如果每次启动程序都重新计算全部页面向量，会浪费时间，因此这个文件负责把已经生成的页面向量保存到磁盘。它将页面编号与向量写入 page_vectors.json，同时用 index_metadata.json 记录向量维度、页面数量和编码配置；再次运行时可以直接加载这些数据，恢复页面索引。"),
    mixed_paragraph("generator.py：", "检索器找出候选页面后，需要从这些页面中确定答案。这个文件先判断 Query 询问的是金额、时间、比例还是其他字段，再查看 Top-K 页面的 facts，根据检索分数和文字匹配程度选择最可信的值，并返回答案、证据页和置信度。它实际上是在预设字段中做规则匹配，所以结果虽然看起来像大模型回答，但没有调用任何生成式模型。"),
    mixed_paragraph("pipeline.py：", "这个文件把前面分散的步骤串成一次完整运行：先准备页面和 Query，再创建编码器、建立页面索引，随后逐条检索问题、生成答案并记录耗时，最后把标准答案、预测答案、证据页和 Top-K 命中结果统一写入 retrieval_results.json，同时计算检索与问答指标。可以把它理解为整个 Demo 的主干流程。"),
    mixed_paragraph("training.py：", "为了比较不同向量融合方式，项目会尝试多组隐藏层权重。这个文件先在训练数据上计算 Query 与正确页面之间的 InfoNCE Loss，再到验证集上比较 MRR 和 Recall，最终保留表现最好的一组配置，并把选择过程写入 training_log.csv。这里没有训练真实神经网络，只是在有限的候选权重中进行搜索。"),
    mixed_paragraph("baselines.py：", "评估一个方案时需要有对照对象，因此这个文件用同一批问题模拟四条检索问答链路。OCR-RAG 会故意把部分字符识别错误或删除字段，SigLIP 方案只使用页面标题，ColPali 方案额外加入版式标记，再与主方案比较检索和回答结果。这些名称表示实验思路，不代表程序真正加载了相应模型。"),
    mixed_paragraph("metrics.py：", "这个文件负责判断系统到底有没有找对和答对。Recall@K 检查正确证据页是否进入前 K 名，MRR 会根据正确页面出现的位置给分，页面越靠前得分越高；EM 和 Accuracy 则把预测答案与标准答案进行比较。检索指标和答案指标分开计算，可以看出错误发生在找页面阶段还是回答阶段。"),
    mixed_paragraph("workflows.py：", "底层文件各自只完成一项任务，而这个文件把它们组织成数据构建、索引构建、检索训练、统一评测和 Demo 五套工作流。它会检查上一步需要的数据是否存在，准备输入目录，调用相应模块，并把每一步结果交给命令行入口，因此承担了项目内部的流程调度工作。"),
    mixed_paragraph("cli.py：", "用户运行项目时不需要逐个调用底层函数，而是通过这个文件选择要执行的阶段。它接收 build-data、build-index、train、evaluate、demo 和 all 等命令，再转交给 workflows.py；其中 all 会按正确顺序连续执行全部流程。"),
    mixed_paragraph("config.py：", "不同实验可能需要改变 Top-K、向量维度、温度系数、训练轮数或数据集比例。这个文件集中读取 config.yaml，把文本形式的参数整理成程序可以直接使用的配置，避免在多个代码文件中重复修改数值。"),
    mixed_paragraph("logging_utils.py：", "项目运行过程中，这个文件会把当前执行阶段、生成文件和指标等信息同时输出到控制台与 logs/pipeline.log。出现异常或结果不符合预期时，可以通过日志确认程序执行到了哪一步。"),
    mixed_paragraph("__init__.py：", "这个文件让 src/vlm_rag 能够作为 Python 包被 scripts 中的程序导入，并保存包的基本信息。它不参与数据处理和检索，但属于 Python 项目组织结构的一部分。"),
    section("5  执行脚本与数据产物"),
    mixed_paragraph("build_dataset.py：", "启动数据构建流程，生成模拟页面、问答数据以及训练集、验证集和测试集，是完整实验的第一步。"),
    mixed_paragraph("build_index.py：", "在页面数据已经生成后运行，它读取 pages.json，调用编码器计算每页向量，并把结果保存到 indexes 目录。"),
    mixed_paragraph("train_retriever.py：", "在索引和数据划分准备完成后运行，用训练集与验证集搜索更合适的检索权重，并保存最佳配置和训练日志。"),
    mixed_paragraph("evaluate.py：", "使用相同问题运行主方案和三个模拟基线，汇总页面检索与答案准确率，最终生成便于比较的 metrics_report.csv。"),
    mixed_paragraph("run_demo.py：", "直接执行一次完整的页面检索问答，把每个问题的预测答案、证据页、置信度和 Top-K 命中页面保存到 retrieval_results.json。"),
    mixed_paragraph("vlm_rag_cli.py：", "提供统一启动方式，可以只运行某一个阶段，也可以使用 all 参数依次完成数据、索引、训练、评测和演示。"),
]


document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="850" w:right="1134" w:bottom="850" w:left="1134" w:header="420" w:footer="420" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>'''

styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体" w:hAnsi="Times New Roman"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="20" w:line="245" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
</w:styles>'''

content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
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
</Relationships>'''

now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
core_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>VLM-RAG 项目说明（论文版）</dc:title><dc:creator>Codex</dc:creator>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''

app_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>Microsoft Office Word</Application><AppVersion>16.0000</AppVersion><Pages>3</Pages>
</Properties>'''

with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
    archive.writestr("[Content_Types].xml", content_types)
    archive.writestr("_rels/.rels", root_rels)
    archive.writestr("word/document.xml", document_xml)
    archive.writestr("word/styles.xml", styles_xml)
    archive.writestr("word/_rels/document.xml.rels", document_rels)
    archive.writestr("docProps/core.xml", core_xml)
    archive.writestr("docProps/app.xml", app_xml)

print(OUTPUT)
