from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


output = Path(__file__).with_name("VLM-RAG项目数据流.docx")


def paragraph(
    text: str,
    *,
    size: int = 27,
    bold: bool = False,
    color: str = "000000",
    after: int = 45,
    shading: str | None = None,
) -> str:
    ppr = [
        '<w:jc w:val="center"/>',
        f'<w:spacing w:before="0" w:after="{after}" w:line="290" w:lineRule="auto"/>',
    ]
    if shading:
        ppr.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{shading}"/>')
        ppr.append('<w:ind w:left="180" w:right="180"/>')
    rpr = [
        '<w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="微软雅黑" w:hAnsi="Microsoft YaHei"/>',
        f'<w:color w:val="{color}"/>',
        f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>',
    ]
    if bold:
        rpr.append('<w:b/><w:bCs/>')
    return (
        f'<w:p><w:pPr>{"".join(ppr)}</w:pPr><w:r><w:rPr>{"".join(rpr)}</w:rPr>'
        f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
    )


def step(text: str) -> str:
    return paragraph(text, size=26, bold=True, color="17365D", after=35, shading="EAF2F8")


def arrow() -> str:
    return paragraph("↓", size=25, bold=True, color="5B9BD5", after=20)


body = [
    paragraph("VLM-RAG 项目完整数据流", size=40, bold=True, color="17365D", after=150),
    step("PDF / PPT / 企业文档"),
    arrow(),
    step("按页转换成页面图片，并建立 Page 数据"),
    paragraph("page_id、标题、图片路径、页面内容", size=21, color="666666", after=35),
    arrow(),
    step("VLM / 页面编码器将每页转换成页面向量"),
    arrow(),
    step("页面向量保存到向量索引"),
    arrow(),
    step("用户输入 Query，Query 编码器生成问题向量"),
    arrow(),
    step("计算问题向量与页面向量的相似度"),
    arrow(),
    step("检索最相关的 Top-K 证据页面"),
    arrow(),
    step("OCR / VLM / 大模型读取证据页并生成答案"),
    arrow(),
    step("输出：答案 + 证据页 + 置信度"),
]

document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body)}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="720" w:right="1200" w:bottom="720" w:left="1200" w:header="420" w:footer="420" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>'''

styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="微软雅黑" w:hAnsi="Microsoft YaHei"/><w:sz w:val="27"/></w:rPr></w:rPrDefault>
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
  <dc:title>VLM-RAG 项目数据流</dc:title><dc:creator>Codex</dc:creator>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''

app_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>Microsoft Office Word</Application><AppVersion>16.0000</AppVersion><Pages>1</Pages>
</Properties>'''

with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    archive.writestr("[Content_Types].xml", content_types)
    archive.writestr("_rels/.rels", root_rels)
    archive.writestr("word/document.xml", document_xml)
    archive.writestr("word/styles.xml", styles_xml)
    archive.writestr("word/_rels/document.xml.rels", document_rels)
    archive.writestr("docProps/core.xml", core_xml)
    archive.writestr("docProps/app.xml", app_xml)

print(output)
