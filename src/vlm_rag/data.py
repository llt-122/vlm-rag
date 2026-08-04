from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Page:
    page_id: str
    doc_id: str
    doc_type: str
    page_no: int
    title: str
    visual_text: str
    layout: str
    facts: dict[str, str]
    image_path: str


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    answer: str
    positive_page_ids: list[str]
    doc_type: str


def build_sample_pages(output_dir: Path, path_root: Path | None = None) -> list[Page]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_pages = _raw_pages()

    pages: list[Page] = []
    for index, item in enumerate(raw_pages, start=1):
        page_id = f"page_{index:03d}"
        image_path = output_dir / f"{page_id}.svg"
        _write_svg_page(image_path, item["title"], item["visual_text"], item["layout"])
        display_path = _display_path(image_path, path_root)
        pages.append(
            Page(
                page_id=page_id,
                doc_id=item["doc_id"],
                doc_type=item["doc_type"],
                page_no=item["page_no"],
                title=item["title"],
                visual_text=item["visual_text"],
                layout=item["layout"],
                facts=item["facts"],
                image_path=display_path,
            )
        )
    return pages


def build_sample_queries() -> list[Query]:
    query_specs = [
        ("采购合同的服务期限是什么？", "服务期限", "contract", 1),
        ("采购合同首付款比例是多少？", "首付款比例", "contract", 1),
        ("采购合同的验收周期是几天？", "验收周期", "contract", 7),
        ("框架协议的违约金比例是多少？", "违约金比例", "contract", 13),
        ("季度经营报表里 Q3 收入是多少？", "Q3收入", "report", 2),
        ("季度经营报表毛利率是多少？", "毛利率", "report", 2),
        ("现金流报表的经营现金流是多少？", "经营现金流", "report", 8),
        ("库存周转报表中周转天数是多少？", "库存周转天数", "report", 14),
        ("销售复盘中哪个区域排名第一？", "销售冠军区域", "ppt", 3),
        ("销售复盘中华东销售额是多少？", "华东销售额", "ppt", 3),
        ("新品发布页的核心卖点是什么？", "核心卖点", "ppt", 9),
        ("渠道复盘的重点渠道是什么？", "重点渠道", "ppt", 15),
        ("李雷这次差旅报销合计多少钱？", "报销合计", "invoice", 4),
        ("李雷报销单的申请人是谁？", "申请人", "invoice", 4),
        ("培训费用单的审批状态是什么？", "审批状态", "invoice", 10),
        ("采购付款单的付款金额是多少？", "付款金额", "invoice", 16),
        ("设备 E17 告警代表什么？", "E17含义", "manual", 5),
        ("设备 E17 告警如何处理？", "E17处理", "manual", 5),
        ("网络故障 N09 的处理步骤是什么？", "N09处理", "manual", 11),
        ("电池告警 B03 的含义是什么？", "B03含义", "manual", 17),
        ("工龄 5 到 10 年年假是多少天？", "5到10年年假", "policy", 6),
        ("工龄 10 年以上年假是多少天？", "10年以上年假", "policy", 6),
        ("加班调休的有效期是多久？", "调休有效期", "policy", 12),
        ("远程办公每周最多几天？", "远程办公上限", "policy", 18),
    ]

    pages = _raw_pages()
    queries: list[Query] = []
    for index, (text, fact_key, doc_type, page_number) in enumerate(query_specs, start=1):
        page = pages[page_number - 1]
        queries.append(
            Query(
                query_id=f"q{index:03d}",
                text=text,
                answer=page["facts"][fact_key],
                positive_page_ids=[f"page_{page_number:03d}"],
                doc_type=doc_type,
            )
        )
    return queries


def load_pages(path: Path) -> list[Page]:
    raw_pages = json.loads(path.read_text(encoding="utf-8"))
    return [Page(**item) for item in raw_pages]


def load_queries(path: Path) -> list[Query]:
    raw_queries = json.loads(path.read_text(encoding="utf-8"))
    return [Query(**item) for item in raw_queries]


def save_dataset(pages: Iterable[Page], queries: Iterable[Query], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pages.json").write_text(
        json.dumps([asdict(page) for page in pages], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "queries.json").write_text(
        json.dumps([asdict(query) for query in queries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _raw_pages() -> list[dict[str, object]]:
    return [
        {
            "doc_id": "contract_001",
            "doc_type": "contract",
            "page_no": 1,
            "title": "采购合同 - 服务期限与付款",
            "visual_text": "甲方 智造集团 乙方 云启科技 服务期限 2026-01-01 至 2026-12-31 付款比例 首付款30 尾款70",
            "layout": "two-column contract clauses with signature block",
            "facts": {"服务期限": "2026-01-01 至 2026-12-31", "首付款比例": "30%"},
        },
        {
            "doc_id": "report_001",
            "doc_type": "report",
            "page_no": 1,
            "title": "季度经营报表 - 收入图表",
            "visual_text": "Q1 收入 1200万 Q2 收入 1580万 Q3 收入 1760万 毛利率 38 图表 折线增长",
            "layout": "line chart above financial summary table",
            "facts": {"Q3收入": "1760万", "毛利率": "38%"},
        },
        {
            "doc_id": "ppt_001",
            "doc_type": "ppt",
            "page_no": 1,
            "title": "销售复盘 PPT - 区域排名",
            "visual_text": "华东 42 华南 35 华北 28 西南 19 区域销售冠军 华东 柱状图 排名",
            "layout": "bar chart with highlighted east region",
            "facts": {"销售冠军区域": "华东", "华东销售额": "42"},
        },
        {
            "doc_id": "invoice_001",
            "doc_type": "invoice",
            "page_no": 1,
            "title": "费用报销单 - 差旅",
            "visual_text": "申请人 李雷 部门 市场部 交通费 2300 住宿费 1800 合计 4100 审批通过",
            "layout": "expense table with approval stamp",
            "facts": {"报销合计": "4100", "申请人": "李雷"},
        },
        {
            "doc_id": "manual_001",
            "doc_type": "manual",
            "page_no": 1,
            "title": "设备巡检手册 - 告警处理",
            "visual_text": "告警代码 E17 含义 温度过高 处理步骤 停机 检查风扇 清理滤网 重启",
            "layout": "flow steps with warning icon",
            "facts": {"E17含义": "温度过高", "E17处理": "停机、检查风扇、清理滤网、重启"},
        },
        {
            "doc_id": "policy_001",
            "doc_type": "policy",
            "page_no": 1,
            "title": "人事制度 - 年假规则",
            "visual_text": "工龄 1到5年 年假5天 工龄 5到10年 年假10天 工龄10年以上 年假15天",
            "layout": "policy table with three seniority rows",
            "facts": {"5到10年年假": "10天", "10年以上年假": "15天"},
        },
        {
            "doc_id": "contract_002",
            "doc_type": "contract",
            "page_no": 2,
            "title": "采购合同 - 交付验收",
            "visual_text": "交付物 数据看板 API接口 培训材料 验收周期 15天 逾期整改 5个工作日 负责人 王敏",
            "layout": "milestone table with acceptance checklist",
            "facts": {"验收周期": "15天", "整改周期": "5个工作日"},
        },
        {
            "doc_id": "report_002",
            "doc_type": "report",
            "page_no": 1,
            "title": "现金流报表 - 经营活动",
            "visual_text": "经营现金流 860万 投资现金流 -240万 筹资现金流 120万 期末现金余额 2140万",
            "layout": "stacked cash-flow table with subtotal row",
            "facts": {"经营现金流": "860万", "期末现金余额": "2140万"},
        },
        {
            "doc_id": "ppt_002",
            "doc_type": "ppt",
            "page_no": 3,
            "title": "新品发布 PPT - 核心卖点",
            "visual_text": "新品 X-Pro 核心卖点 低功耗 高稳定 边缘推理 客户价值 降本20 响应速度提升35",
            "layout": "three feature cards with product image area",
            "facts": {"核心卖点": "低功耗、高稳定、边缘推理", "降本比例": "20%"},
        },
        {
            "doc_id": "invoice_002",
            "doc_type": "invoice",
            "page_no": 1,
            "title": "培训费用报销单 - 外部课程",
            "visual_text": "申请人 韩梅梅 部门 研发部 课程费 6800 资料费 300 合计 7100 审批状态 待复核",
            "layout": "expense rows with pending approval badge",
            "facts": {"培训合计": "7100", "审批状态": "待复核"},
        },
        {
            "doc_id": "manual_002",
            "doc_type": "manual",
            "page_no": 4,
            "title": "网络设备手册 - N09 故障",
            "visual_text": "故障代码 N09 含义 网络链路异常 处理步骤 检查网线 重置网关 查看交换机端口",
            "layout": "network topology diagram with numbered steps",
            "facts": {"N09含义": "网络链路异常", "N09处理": "检查网线、重置网关、查看交换机端口"},
        },
        {
            "doc_id": "policy_002",
            "doc_type": "policy",
            "page_no": 2,
            "title": "人事制度 - 加班调休",
            "visual_text": "工作日加班 1比1 调休 周末加班 1比2 调休 调休有效期 90天 逾期自动清零",
            "layout": "policy comparison table with notes",
            "facts": {"调休有效期": "90天", "周末加班调休": "1比2"},
        },
        {
            "doc_id": "contract_003",
            "doc_type": "contract",
            "page_no": 3,
            "title": "框架协议 - 违约责任",
            "visual_text": "违约责任 延迟交付 每日千分之三 违约金比例 5 合同上限 20 争议解决 上海仲裁",
            "layout": "dense legal clauses with risk highlight",
            "facts": {"违约金比例": "5%", "争议解决": "上海仲裁"},
        },
        {
            "doc_id": "report_003",
            "doc_type": "report",
            "page_no": 2,
            "title": "库存周转报表 - 品类明细",
            "visual_text": "A类库存 3200 B类库存 2100 C类库存 900 库存周转天数 43 呆滞库存占比 8",
            "layout": "inventory table with pie chart",
            "facts": {"库存周转天数": "43天", "呆滞库存占比": "8%"},
        },
        {
            "doc_id": "ppt_003",
            "doc_type": "ppt",
            "page_no": 5,
            "title": "渠道复盘 PPT - 渠道贡献",
            "visual_text": "直营 52 代理 31 电商 17 重点渠道 直营 渠道策略 提升代理覆盖 优化电商转化",
            "layout": "donut chart with strategy callouts",
            "facts": {"重点渠道": "直营", "代理贡献": "31"},
        },
        {
            "doc_id": "invoice_003",
            "doc_type": "invoice",
            "page_no": 1,
            "title": "采购付款单 - 服务器",
            "visual_text": "供应商 星河硬件 付款金额 128000 付款方式 银行转账 发票状态 已收到 审批人 陈晨",
            "layout": "payment form with invoice status stamp",
            "facts": {"付款金额": "128000", "付款方式": "银行转账"},
        },
        {
            "doc_id": "manual_003",
            "doc_type": "manual",
            "page_no": 7,
            "title": "电池维护手册 - B03 告警",
            "visual_text": "告警代码 B03 含义 电池电压偏低 处理步骤 接入备用电源 检查电池组 更换老化模块",
            "layout": "battery icon with maintenance flow",
            "facts": {"B03含义": "电池电压偏低", "B03处理": "接入备用电源、检查电池组、更换老化模块"},
        },
        {
            "doc_id": "policy_003",
            "doc_type": "policy",
            "page_no": 4,
            "title": "办公制度 - 远程办公",
            "visual_text": "远程办公 需提前申请 每周最多 2天 涉密岗位 不适用 审批人 直属经理",
            "layout": "rule cards with eligibility matrix",
            "facts": {"远程办公上限": "每周2天", "审批人": "直属经理"},
        },
    ]


def _write_svg_page(path: Path, title: str, visual_text: str, layout: str) -> None:
    words = visual_text.split()
    rows = [" ".join(words[index : index + 7]) for index in range(0, len(words), 7)]
    body = "\n".join(
        f'<text x="48" y="{140 + row_index * 38}" class="body">{row}</text>'
        for row_index, row in enumerate(rows)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1200" viewBox="0 0 900 1200">
  <style>
    .page {{ fill: #fbfaf7; }}
    .title {{ font: 700 34px sans-serif; fill: #1f2937; }}
    .body {{ font: 24px sans-serif; fill: #374151; }}
    .layout {{ font: 20px sans-serif; fill: #6b7280; }}
    .panel {{ fill: #ffffff; stroke: #d1d5db; stroke-width: 2; }}
    .accent {{ fill: #2563eb; opacity: 0.18; }}
  </style>
  <rect class="page" width="900" height="1200"/>
  <rect class="accent" x="0" y="0" width="900" height="88"/>
  <text x="48" y="64" class="title">{_escape_xml(title)}</text>
  <rect class="panel" x="40" y="105" width="820" height="420" rx="8"/>
  {body}
  <rect class="panel" x="40" y="580" width="820" height="420" rx="8"/>
  <text x="72" y="635" class="layout">Visual layout: {_escape_xml(layout)}</text>
  <line x1="90" y1="910" x2="790" y2="720" stroke="#2563eb" stroke-width="12" opacity="0.5"/>
  <circle cx="180" cy="885" r="32" fill="#16a34a" opacity="0.72"/>
  <circle cx="430" cy="815" r="32" fill="#f59e0b" opacity="0.72"/>
  <circle cx="690" cy="745" r="32" fill="#ef4444" opacity="0.72"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _display_path(path: Path, path_root: Path | None) -> str:
    if path_root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(path_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
