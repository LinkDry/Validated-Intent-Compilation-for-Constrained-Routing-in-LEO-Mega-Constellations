#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
PROJECT_ROOT = REPO_ROOT / "projects" / "leo-intent-routing"
CODE_ROOT = PROJECT_ROOT / "workspace" / "code"
OUTPUT_ROOT = CODE_ROOT / "output"
PAPER_ROOT = PROJECT_ROOT / "workspace" / "paper"
OUTPUT_DIR = REPO_ROOT / "output" / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "leo-intent-routing-project-evolution-report-2026-04-09.pdf"


def register_fonts() -> None:
    candidates = [
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont("ReportFont", str(font_path)))
            return
    raise FileNotFoundError("No suitable Chinese font found under C:/Windows/Fonts.")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(html.escape(text).replace("\n", "<br/>"), style)


def rich(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def format_pct(value: float, digits: int = 1, fraction: bool = True) -> str:
    number = value * 100 if fraction and value <= 1.0 + 1e-9 else value
    return f"{number:.{digits}f}%"


def format_ms(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f} ms"


def format_seconds_from_ms(value: float, digits: int = 1) -> str:
    return f"{value / 1000:.{digits}f} s"


def line_count(path: Path) -> int:
    return len(read_text(path).splitlines())


def parse_paper_log(path: Path) -> dict[str, int]:
    text = read_text(path)
    match = re.search(r"Output written on .* \((\d+) pages, (\d+) bytes\)", text)
    pages = int(match.group(1)) if match else -1
    size_bytes = int(match.group(2)) if match else -1
    return {
        "pages": pages,
        "bytes": size_bytes,
        "overfull": len(re.findall(r"Overfull \\hbox", text)),
        "underfull": len(re.findall(r"Underfull \\hbox", text)),
        "fatal_errors": len(re.findall(r"^! ", text, flags=re.MULTILINE)),
    }


def extract_state_claim(state_text: str) -> str:
    match = re.search(r'next_action:\s*"([^"]+)"', state_text)
    return match.group(1) if match else "STATE.md 中未解析到 next_action。"


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=sample["Title"],
            fontName="ReportFont",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#102542"),
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=sample["Normal"],
            fontName="ReportFont",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#355070"),
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=sample["Heading1"],
            fontName="ReportFont",
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#0b3954"),
            spaceBefore=8,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=sample["Heading2"],
            fontName="ReportFont",
            fontSize=12.5,
            leading=18,
            textColor=colors.HexColor("#114b5f"),
            spaceBefore=7,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=sample["Heading3"],
            fontName="ReportFont",
            fontSize=10.8,
            leading=15,
            textColor=colors.HexColor("#1a5872"),
            spaceBefore=5,
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=sample["BodyText"],
            fontName="ReportFont",
            fontSize=10.2,
            leading=15,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#222222"),
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "body_center": ParagraphStyle(
            "body_center",
            parent=sample["BodyText"],
            fontName="ReportFont",
            fontSize=10.2,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#222222"),
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            parent=sample["BodyText"],
            fontName="ReportFont",
            fontSize=8.8,
            leading=12,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#444444"),
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=sample["BodyText"],
            fontName="ReportFont",
            fontSize=10.0,
            leading=14,
            alignment=TA_LEFT,
            leftIndent=10,
            firstLineIndent=-10,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "box_title": ParagraphStyle(
            "box_title",
            parent=sample["Heading3"],
            fontName="ReportFont",
            fontSize=10.5,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "box_body": ParagraphStyle(
            "box_body",
            parent=sample["BodyText"],
            fontName="ReportFont",
            fontSize=9.0,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#13293d"),
            wordWrap="CJK",
        ),
    }


def add_paragraphs(story: list, styles: dict[str, ParagraphStyle], text: str) -> None:
    blocks = [block.strip() for block in text.strip().split("\n\n") if block.strip()]
    for block in blocks:
        story.append(paragraph(block, styles["body"]))


def add_bullets(story: list, styles: dict[str, ParagraphStyle], items: list[str]) -> None:
    for item in items:
        story.append(paragraph(f"- {item}", styles["bullet"]))


def info_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], widths) -> Table:
    data = [[paragraph(a, styles["small"]), paragraph(b, styles["small"])] for a, b in rows]
    table = Table(data, colWidths=widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fbff")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#9bb1c8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c9d7e3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def colored_table(data: list[list[str]], styles: dict[str, ParagraphStyle], widths, header_bg: str = "#dbe9f4") -> Table:
    table_data = []
    for row_idx, row in enumerate(data):
        style = styles["body"] if row_idx == 0 else styles["small"]
        table_data.append([paragraph(str(cell), style) for cell in row])
    table = Table(table_data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12263a")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#8aa2b2")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c0ced9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def flow_box(title: str, body: str, styles: dict[str, ParagraphStyle], width: float) -> Table:
    table = Table(
        [[paragraph(title, styles["box_title"])], [paragraph(body, styles["box_body"])]],
        colWidths=[width],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#e8f1f8")),
                ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#5f7c8a")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#5f7c8a")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("ReportFont", 8)
    canvas.setFillColor(colors.HexColor("#5b6770"))
    canvas.drawString(doc.leftMargin, 10 * mm, "leo-intent-routing 项目演化与论文全景报告")
    canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def gather_context() -> dict:
    state_text = read_text(PROJECT_ROOT / "STATE.md")
    context = {
        "state_text": state_text,
        "state_claim": extract_state_claim(state_text),
        "review_state": load_json(PROJECT_ROOT / "review-state.json"),
        "dashboard_data": load_json(PROJECT_ROOT / "workspace" / "dashboard-data.json"),
        "ablation_full": load_json(OUTPUT_ROOT / "ablation_full.json"),
        "ablation_no_repair": load_json(OUTPUT_ROOT / "ablation_no_repair.json"),
        "ablation_zero_shot": load_json(OUTPUT_ROOT / "ablation_zero_shot.json"),
        "ablation_rule_based": load_json(OUTPUT_ROOT / "ablation_rule_based.json"),
        "benchmark_eval": load_json(OUTPUT_ROOT / "benchmark_eval_240.json"),
        "ood_eval": load_json(OUTPUT_ROOT / "ood_eval_results.json"),
        "confusion": load_json(OUTPUT_ROOT / "verifier_confusion_matrix.json"),
        "runtime": load_json(OUTPUT_ROOT / "pass8_runtime.json"),
        "reachability": load_json(OUTPUT_ROOT / "reachability_separation.json"),
        "cross_constellation": load_json(OUTPUT_ROOT / "cross_constellation_gnn.json"),
        "polar_exclusion": load_json(OUTPUT_ROOT / "polar_exclusion_gnn.json"),
        "adversarial": load_json(OUTPUT_ROOT / "adversarial_safety.json"),
        "independent_oracle": load_json(OUTPUT_ROOT / "independent_oracle_32.json"),
        "topology_sweep": load_json(OUTPUT_ROOT / "topology_sweep.json"),
        "e2e": load_json(OUTPUT_ROOT / "e2e_eval_results.json"),
        "model_4b": load_json(OUTPUT_ROOT / "ablation_model_4b.json"),
        "paper_log": parse_paper_log(PAPER_ROOT / "paper_main.log"),
        "paper_lines_intermediate": line_count(OUTPUT_ROOT / "paper_main.tex"),
        "paper_lines_final": line_count(PAPER_ROOT / "paper_main.tex"),
        "paper_tables_intermediate": line_count(OUTPUT_ROOT / "paper_new_tables.tex"),
        "paper_tables_final": line_count(PAPER_ROOT / "paper_new_tables.tex"),
        "final_submission_exists": (PAPER_ROOT / "arxiv-submission.tar.gz").exists(),
    }
    return context


def build_cover(story: list, styles: dict[str, ParagraphStyle], ctx: dict) -> None:
    full = ctx["ablation_full"]
    runtime = ctx["runtime"]["all"]
    paper_log = ctx["paper_log"]
    story.append(Spacer(1, 22))
    story.append(paragraph("leo-intent-routing 项目演化与论文全景报告", styles["title"]))
    story.append(paragraph("面向非本领域读者的中文深度复盘、技术教学与投稿审视", styles["subtitle"]))
    story.append(paragraph("版本日期：2026-04-09", styles["subtitle"]))
    story.append(Spacer(1, 10))

    overview_rows = [
        ["一句话定位", "这不是“让 LLM 直接控制卫星路由”的论文，而是一条“自然语言意图 -> 约束程序 -> 确定性验证 -> GNN/Dijkstra 路由”的安全系统路线。"],
        ["为什么比 satcom-llm-drl 顺利", "因为项目一开始就吸收了父项目的失败经验，主动把 LLM 从在线连续数值控制中移出，转而负责离线语义编译。"],
        ["最终最可信的主张", f"GNN 路由在训练星座上达到 99.8% PDR、17x 加速；LLM 编译器在 240 条意图上做到 {format_pct(full['compiled_rate'])} 编译率与 {format_pct(full['full_match_rate'])} 全语义匹配；8-pass 验证器把 infeasible intent 的 unsafe acceptance 压到 0%。"],
        ["当前投稿态的重要提醒", f"STATE.md 自述为“10pp, 0 errors”，但当前 paper_main.log 实际记录为 {paper_log['pages']} 页、{paper_log['fatal_errors']} 个致命错误、{paper_log['overfull']} 个 overfull、{paper_log['underfull']} 个 underfull。"],
        ["报告目标", "帮助你不仅知道项目做了什么，还理解这些技术概念为什么存在、每次实验为什么要做、哪些结论值得信，哪些地方仍需谨慎。"],
    ]
    story.append(info_table(overview_rows, styles, [33 * mm, 131 * mm]))
    story.append(Spacer(1, 8))

    story.append(paragraph("这份报告怎么读", styles["h1"]))
    add_bullets(
        story,
        styles,
        [
            "先读第 1 到第 3 节，快速建立“项目究竟是什么”的总图景。",
            "再读第 4 节时间线，它解释了这个项目为什么能一步步收敛，而不是像父项目那样长期陷在方法级不稳定里。",
            "第 5 节是技术教学区，面向不熟悉卫星网络、GNN、约束编译、验证器的人写成。",
            "第 6 到第 7 节重点讲论文写作演化与全面审视，包括证据一致性、审稿风险、最强结论与尚未闭合的问题。",
            "最后一节给出文件导读，你可以按路径回到仓库里逐项核对本报告的判断。",
        ],
    )
    


def build_summary_section(story: list, styles: dict[str, ParagraphStyle], ctx: dict) -> None:
    full = ctx["ablation_full"]
    ood = ctx["ood_eval"]
    safety = ctx["confusion"]["safety"]
    runtime = ctx["runtime"]["all"]
    story.append(paragraph("1. 这篇论文真正是什么", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        """
        如果只用最直白的话来概括，`leo-intent-routing` 的主张不是“LLM 已经学会了卫星路由”，而是“可以把 LLM 放在它擅长的语义编译位置，用确定性验证器和传统图算法把安全边界兜住，再由 GNN 或 Dijkstra 去做真正的路由决策”。

        这个定位非常关键。因为它决定了项目从一开始就不是在赌一个端到端的神奇模型，而是在设计一个分层系统：自然语言负责表达意图，ConstraintProgram 负责承接意图，Verifier 负责拦截危险或不可行程序，Grounding 负责把约束投影到图结构，Router 负责在受约束的图上算路。

        这条主线之所以成立，是因为父项目已经把相反路线的风险讲清楚了：一旦让 LLM 直接做在线 reward 或连续数值控制，系统的主要难点会从“懂不懂语义”变成“输出稳定不稳定”，而这个问题会直接破坏训练与控制闭环。
        """,
    )

    summary_rows = [
        ["Phase A 路由层", "3 层 GAT + MLP cost-to-go 头；152,193 参数；监督蒸馏自 Dijkstra；99.8% PDR；17x 推理加速。"],
        ["Phase B 编译层", f"Qwen3.5-9B + 6-shot prompting + repair loop；240 意图基准上 full pipeline 达到 {format_pct(full['compiled_rate'])} compiled、{format_pct(full['full_match_rate'])} full match。"],
        ["安全层", f"8-pass deterministic validator + Pass 8 feasibility certifier；unsafe acceptance={format_pct(safety['infeasible_unsafe_accept_rate'])}；运行时中位数 {format_ms(runtime['median_ms'])}。"],
        ["泛化层", f"OOD 改写意图上 compiled={format_pct(ood['compiled_rate'])}，scorable 样本 full match={format_pct(ood['full_match_rate'])}；说明语言泛化尚可，但复杂组合仍是难点。"],
        ["系统层结论", "最稳的论文口径不是“GNN 超越 Dijkstra”或“LLM 直接控网”，而是“受验证的意图编译系统，让自然语言网络意图可以安全落地到受约束的 LEO 路由”。"],
    ]
    story.append(info_table(summary_rows, styles, [34 * mm, 130 * mm]))
    story.append(Spacer(1, 8))

    story.append(paragraph("2. 为什么这个项目比父项目更顺", styles["h1"]))
    add_bullets(
        story,
        styles,
        [
            "它从父项目继承的不是代码，而是失败知识：LLM 不适合在线连续数值输出，PPO 对 reward non-stationarity 极敏感，多 seed 和 held-out 测试必须前置。",
            "它把系统拆成了可局部修复的模块。GNN 不行就改蒸馏架构，验证器有漏洞就加 Pass 8，而不是整套训练范式一起崩掉。",
            "它的 claim 更克制。路由层只声称“匹配 Dijkstra + 加速”，编译层只声称“把自然语言变成约束程序”，验证层只声称“把不安全程序挡在路由层之外”。",
            "它从很早就把评估做成“工程证据链”，而不是只盯单轮漂亮分数：ablation、OOD、e2e、topology sweep、confusion matrix、independent oracle 都是这条思路的产物。",
        ],
    )
    


def build_evidence_section(story: list, styles: dict[str, ParagraphStyle], ctx: dict) -> None:
    paper_log = ctx["paper_log"]
    review_state = ctx["review_state"]
    dashboard = ctx["dashboard_data"]
    story.append(paragraph("3. 证据地图与可信度边界", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        """
        这份报告不是只复述论文摘要，而是把仓库中的不同证据层拆开看。对这个项目来说，最重要的一个认知是：不是所有文件都处在同一个时间点，也不是所有文件都适合直接拿来代表最终结论。

        最值得信赖的“最终投稿态”证据在 `projects/leo-intent-routing/workspace/paper/`。其中 `paper_main.tex`、`paper_new_tables.tex`、`paper_main.log` 和 `arxiv-submission.tar.gz` 代表的是收束后的论文版本。与之相对，`workspace/code/output/` 更像实验推进中的中间态，它很重要，但不能不加区分地直接等同于投稿版。

        另外，`review-state.json` 与 `workspace/dashboard-data.json` 反映的是较早的冻结瞬间。它们能帮助我们恢复“何时进入写作阶段”，但不能覆盖 4 月 8 日后所有补实验和口径收敛动作。因此本报告会显式标注：哪些结论来自前期快照，哪些来自后期最终稿。
        """,
    )

    evidence_rows = [
        ["项目元信息", "project-brief.md、experiment-memory.md、STATE.md。用来恢复项目的 lineage、阶段划分、active idea 和后期自我总结。"],
        ["阶段冻结快照", f"review-state.json 记录 2026-04-07 22:30 之前的 9 步冻结点；dashboard-data.json 记录 2026-04-08 11:15 已经进入 paper draft v1。"],
        ["中间实验态", "workspace/code/output/*.json 与 workspace/code/output/paper_*.tex。适合追踪实验演化、指标变化和中间写作版本。"],
        ["最终投稿态", f"workspace/paper/paper_main.tex、paper_new_tables.tex、paper_main.log、arxiv-submission.tar.gz。当前最终稿存在={ctx['final_submission_exists']}。"],
        ["目录缺口", "papers/drafts、papers/reviews、archive/artifacts、plans、refine-logs 基本为空，因此“写作过程”需要靠状态文件和 tex 版本间接重建。"],
        ["当前编译现实", f"paper_main.log 记录 {paper_log['pages']} 页、{paper_log['bytes']} bytes、{paper_log['fatal_errors']} 个致命错误、{paper_log['overfull']} 个 overfull、{paper_log['underfull']} 个 underfull。"],
    ]
    story.append(info_table(evidence_rows, styles, [34 * mm, 130 * mm]))
    story.append(Spacer(1, 8))

    snapshot_rows = [
        ["4 月 7 日快照", review_state["summary"]],
        ["4 月 8 日 11:15 仪表盘", dashboard["project"]["next_action"]],
        ["4 月 8 日晚 STATE 自述", ctx["state_claim"]],
        ["需要额外提醒的时间差", "STATE.md 自称“10pp, 0 errors”，但当前 log 仍是 9 页并带 box 警告。这更像写作管理状态与最终文件之间存在半天到一天的时滞，而不是核心实验内容矛盾。"],
    ]
    story.append(info_table(snapshot_rows, styles, [34 * mm, 130 * mm]))
    story.append(PageBreak())


def build_timeline_section(story: list, styles: dict[str, ParagraphStyle], ctx: dict) -> None:
    story.append(paragraph("4. 从 idea 到成稿的完整主时间线", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        """
        下面这张时间线不是简单列实验名，而是回答每一步“当时在解决什么问题、为什么要转向、这一步改变了项目什么”。这也是理解本项目最重要的一节。
        """,
    )

    timeline_rows = [
        ["父项目 lessons -> 本项目立题", "接受 satcom-llm-drl 已证明的失败事实：LLM 不适合在线 reward architect，于是转向“LLM-to-Graph Constraint Compilation”。这一步决定了本项目的边界。"],
        ["Iter 1: bilinear + PPO", "保留父项目残留想法做一次确认性尝试，结果 Max PDR 只有 4.4%。这不是细节没调好，而是再次证明在线 RL 路线不该继续押注。"],
        ["Iter 2: ctg-v1", "从 PPO pivot 到监督蒸馏，但第一版 cost-to-go 头仍失败，acc 只有 24.1%。失败原因是 softplus 与低秩双线性结构太弱。"],
        ["Iter 3-4: ctg-v2 / ctg-v3-final", "换成 MLP scorer、加 bias init、位置特征与 phased loss。正式跑通 Phase A：91.4% next-hop accuracy、99.8% PDR、17x speedup。GNN 作为路由器终于站稳。"],
        ["Iter 5: compiler ablation", "项目真正进入论文主线。Few-shot + verifier + repair loop 让自然语言意图编译变得可量化，full pipeline 达到 97.9% compiled / 86.2% full match。"],
        ["Iter 6-9: e2e / audit / OOD / topology sweep", "证明系统不是单点 demo：端到端零违规、corruption audit 100% 检出、OOD 改写仍可用、退化拓扑下 GNN 与 Dijkstra 一致。4 月 7 日晚由此进入第一轮写作冻结。"],
        ["Iter 10: confusion matrix 危机", "一张 3-way confusion matrix 暴露致命安全洞：旧 validator 在 infeasible intents 上可能出现高 unsafe acceptance。项目进入真正的“审稿人视角补洞阶段”。"],
        ["Iter 11-14: reachability / rule-based / adversarial / draft v1", "一边补写论文草稿，一边解释极区场景 raw PDR 差距、一边证明 LLM 相比 rule-based 的价值、一边用 adversarial 测试找安全空洞。"],
        ["Iter 15: Pass 8 feasibility certifier", "关键补丁。把“结构正确但物理不可行”的漏洞交给构造性图算法认证，unsafe acceptance 从 72% 降到 0%，对抗安全从 73.3% 提到 100%。"],
        ["Iter 16: phase-b-paper-updates", "补齐 runtime、cross-constellation、polar exclusion、independent oracle，并把论文更新到 8-pass 最终叙事。此时 Phase B 被自述为 COMPLETE。"],
        ["最终投稿态", "最终稿把系统 framing 收束为“Intent Compiler thesis + end-to-end system presentation”，提交包位于 workspace/paper。真正被论文保留下来的，是验证过的系统链，而不是所有中间实验尝试。"],
    ]
    long_table = LongTable(
        [[paragraph(a, styles["small"]), paragraph(b, styles["small"])] for a, b in [["阶段", "核心变化"]] + timeline_rows],
        colWidths=[42 * mm, 122 * mm],
        repeatRows=1,
    )
    long_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceaf7")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#8aa2b2")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c0ced9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(long_table)
    story.append(Spacer(1, 8))

    story.append(paragraph("这一时间线带来的总判断", styles["h2"]))
    add_bullets(
        story,
        styles,
        [
            "本项目真正的成功不是“从没失败”，而是每一次失败都迫使系统边界更清晰。",
            "Phase A 的成功让路由层不再是问题中心，Phase B 的危机则把注意力转移到编译正确性与安全验证上。",
            "最后成稿保留下来的科学资产，不是所有实验都成功，而是哪些失败被解释清楚、哪些漏洞被工程化修复、哪些 claim 因证据不足而主动收缩。",
        ],
    )
    story.append(PageBreak())


def build_teaching_section(story: list, styles: dict[str, ParagraphStyle], ctx: dict) -> None:
    story.append(paragraph("5. 技术教学：把这篇论文真正讲明白", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        """
        这一节面向“我基本不懂该方向”的读者来写。目标不是把所有公式推一遍，而是建立足够牢固的概念框架，让你回头看代码、表格和论文时知道每个模块在干什么、为什么存在、做不到什么。
        """,
    )

    story.append(paragraph("5.1 LEO 星座路由到底难在哪里", styles["h2"]))
    add_paragraphs(
        story,
        styles,
        """
        LEO 指低轨卫星星座。和地面网络相比，LEO 路由的核心难点是“网络图本身在动”。卫星不断绕地球飞行，星间链路长度会变，高纬区域还会因为极区拓扑和链路条件出现掉边或可达性骤变。于是，路由器面临的不是一张静态图，而是一张持续变化的时变图。

        对运维人员来说，他们提出的往往不是“把 A 的下一跳改成 B”这种底层配置，而是“禁用 5 号轨道面”“避开 45 度以上极区链路”“金融业务时延必须低于 80 ms”这种高层目标。论文的系统意义就在这里：它试图把这种自然语言运维意图，可靠地翻译成底层图约束，并在真正下发前完成安全检查。
        """,
    )

    story.append(paragraph("5.2 整套系统的五步流水线", styles["h2"]))
    flow_width = 150 * mm
    story.append(flow_box("步骤 1：自然语言意图", "人类输入高层约束，如禁用节点、绕开区域、限制时延、触发条件和回退策略。", styles, flow_width))
    story.append(Spacer(1, 2))
    story.append(paragraph("↓", styles["body_center"]))
    story.append(flow_box("步骤 2：LLM Intent Compiler", "Qwen3.5-9B 根据 few-shot 示例把意图编译成结构化 ConstraintProgram，而不是直接产出路由表。", styles, flow_width))
    story.append(Spacer(1, 2))
    story.append(paragraph("↓", styles["body_center"]))
    story.append(flow_box("步骤 3：Deterministic Verifier", "多 pass 验证结构、实体、数值、冲突、物理可行性与路由可行性。危险程序在此被 reject 或 abstain。", styles, flow_width))
    story.append(Spacer(1, 2))
    story.append(paragraph("↓", styles["body_center"]))
    story.append(flow_box("步骤 4：Constraint Grounding", "把约束程序落到当前拓扑，转换成节点 mask、边 mask、deadline、流量选择器等图层约束。", styles, flow_width))
    story.append(Spacer(1, 2))
    story.append(paragraph("↓", styles["body_center"]))
    story.append(flow_box("步骤 5：GNN / Dijkstra Routing", "在受约束的图上计算下一跳。GNN 负责加速，Dijkstra 负责基线、教师与回退。", styles, flow_width))
    story.append(Spacer(1, 8))

    story.append(paragraph("5.3 关键术语速讲", styles["h2"]))
    glossary_rows = [
        ["GNN cost-to-go", "不是直接猜下一跳，而是估计“从当前节点到目标节点还剩多少代价”。这更接近最短路的动态规划思想，也更容易蒸馏 Dijkstra。"],
        ["Dijkstra baseline", "既是训练教师，也是质量基线，还是系统 fallback。它慢但透明、稳定、可验证。"],
        ["Intent compilation", "把自然语言运维意图编译成结构化程序。论文强调的是“编译”和“验证”，而不是让 LLM 直接给出控制动作。"],
        ["ConstraintProgram", "中间表示层。里面会写清 flow selectors、hard constraints、soft constraints、event conditions、fallback policy 等字段。"],
        ["Deterministic verifier", "同一输入永远得到同一判定，不靠模型感觉。它的职责是安检，不是优化。"],
        ["Pass 8 feasibility certifier", "论文后期最关键的新模块。它不是检查 JSON 是否好看，而是构造性地证明约束在当前图上是否真的做得到。"],
        ["Reachability separation", "把 raw PDR 拆成“图本身可达上限”和“在可达对上的路由质量”，避免误把拓扑天花板当作路由器退化。"],
        ["OOD", "out-of-distribution。这里至少分两类：语言改写 OOD 和拓扑几何 OOD。前者问编译器稳不稳，后者问路由器换星座后还行不行。"],
    ]
    story.append(info_table(glossary_rows, styles, [36 * mm, 128 * mm]))
    story.append(Spacer(1, 8))

    story.append(paragraph("5.4 为什么 GNN 要学 cost-to-go，而不是直接学下一跳", styles["h2"]))
    add_paragraphs(
        story,
        styles,
        """
        直接把问题做成“给定源和目的地，分类出下一跳是谁”当然也可以，但那样容易让模型学成一张僵硬的查表器。cost-to-go 的想法更接近最短路径的本质：如果你能估计“从邻居 j 出发到目标 d 的剩余代价”，那么当前节点只要选代价最小的邻居即可。

        这种设计的好处有三点。第一，它和 Dijkstra 的价值函数天然对齐，所以监督蒸馏更顺。第二，它对不同目的地共享了更多结构知识，不是每个目的地都得重新学一套硬编码规则。第三，它更容易被解释成“学习一个近似最短路势函数”，而不是黑盒分类器。
        """,
    )

    story.append(paragraph("5.5 为什么 LLM 在这里做编译器，而不是做路由器", styles["h2"]))
    add_paragraphs(
        story,
        styles,
        """
        这是整个项目最值得学的设计判断。父项目已经表明，LLM 一旦被放到在线连续数值控制的位置，真正困难的地方就变成了输出波动、训练失稳和 credit assignment 混乱。换句话说，LLM 的自然语言能力并不会自动迁移成稳定的控制器能力。

        但在本项目里，LLM 被放在更合适的位置：读取带组合结构的自然语言约束，把它转成 typed IR。这里的主要难点是语义理解、组合推理和字段对齐，正是大模型相对规则模板更有优势的地方。这也是为什么 rule-based baseline 在 single intents 还能过得去，但在 compositional intents 上会明显落后。
        """,
    )

    story.append(paragraph("5.6 验证器为什么是系统的真正安全核心", styles["h2"]))
    add_paragraphs(
        story,
        styles,
        """
        一个看起来像样的 JSON 程序，并不等于它就能安全下发。它可能引用了不存在的实体，可能约束类型挂错对象，可能数值越界，可能几条约束互相冲突，甚至可能在物理网络上根本不存在满足条件的路径。

        所以 verifier 的价值不是让编译器显得更漂亮，而是在 LLM 和网络执行层之间建立一个“非学习型、可解释、可复现”的边界。论文后期新增的 Pass 8 更是把安全边界从“结构合法”推进到“可执行可行”。这一步是为什么项目后期会把 unsafe acceptance 从 72% 压到 0%。
        """,
    )

    story.append(paragraph("5.7 Pass 8 到底在做什么", styles["h2"]))
    add_bullets(
        story,
        styles,
        [
            "如果只是检查连通性，可用 BFS 找 witness path。",
            "如果有时延约束，可用 Dijkstra 在带权图上做证据构造。",
            "如果有限跳约束，可做 hop-layered 或限跳搜索。",
            "如果要求多条边不相交路径，可用 max-flow/Edmonds-Karp 一类方法。",
            "如果超出了当前支持片段，系统宁可 abstain，也不假装可行。",
        ],
    )
    add_paragraphs(
        story,
        styles,
        """
        注意这意味着 Pass 8 不是万能求解器。它只覆盖论文定义的 certified fragments，因此高 abstain rate 不代表它没工作，而代表作者选择了“有证据才 accept，没有证据宁可保守”的安全策略。
        """,
    )

    story.append(paragraph("5.8 Reachability separation 为什么重要", styles["h2"]))
    add_paragraphs(
        story,
        styles,
        """
        这是本论文很值得借鉴的分析思想。直觉上，如果在 polar avoidance 场景下 GNN 的 PDR 只有 34.6%，而 Dijkstra 有 47.9%，我们会以为 GNN 更差。但 reachability separation 告诉我们，约束一旦切掉了大量极区相关边，真正的问题可能已经不是“路由器会不会选路”，而是“图上本来就只剩下 24% 的源宿对可达”。

        论文进一步证明：在这些仍然可达的 OD 对上，GNN 和 Dijkstra 都能达到 100% reachable PDR。也就是说，表面 13 个百分点的 raw gap，其实主要是 reachability ceiling 造成的，而不是路由质量退化。这让 GNN 的角色从“可能不如 Dijkstra 的近似器”变成“在可达域上表现等价、只是受全局可达上限影响的加速器”。
        """,
    )
def build_writing_section(story: list, styles: dict[str, ParagraphStyle], ctx: dict) -> None:
    story.append(paragraph("6. 论文写作与成稿演化", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        """
        这个项目的写作过程，并不是“实验做完以后把结果往论文里填”。更准确地说，它是“先形成一个足以投稿的 draft v1，再围绕审稿人可能攻击的漏洞持续补实验，最后把论文 framing 主动收紧”。

        由于 `papers/drafts`、`papers/reviews`、`archive/artifacts` 等目录几乎为空，这条写作演化链不能像普通项目那样靠一串连续草稿直接恢复。它主要要靠三个层面的间接证据拼回去：一是 `experiment-memory.md` 中的迭代记录；二是 `review-state.json` 与 `dashboard-data.json` 的阶段冻结快照；三是 `workspace/code/output/` 与 `workspace/paper/` 两套 tex 文件的内容差异。
        """,
    )

    writing_rows = [
        ["早期 framing", "Intent Compiler thesis + end-to-end system presentation。核心不是证明 GNN 胜过 Dijkstra，也不是让 LLM 直接在线控网。"],
        ["draft v1 时间点", "dashboard-data.json 表明 2026-04-08 11:15 左右论文 draft v1 已完成，下一步是 repo 准备与 polish。"],
        ["中间稿规模", f"workspace/code/output/paper_main.tex 约 {ctx['paper_lines_intermediate']} 行；paper_new_tables.tex 约 {ctx['paper_tables_intermediate']} 行。"],
        ["最终稿规模", f"workspace/paper/paper_main.tex 约 {ctx['paper_lines_final']} 行；paper_new_tables.tex 约 {ctx['paper_tables_final']} 行；另有 arxiv-submission.tar.gz。"],
        ["后期补洞主题", "Pass 8 feasibility、3-way confusion matrix、runtime、independent oracle、cross-constellation、polar exclusion、rule-based baseline、reachability separation。"],
        ["最终叙事收束", "把 GNN 明确降级为 optional accelerator，把 validator 明确升级为安全核心，把 abstain 明确解释成保守 fallback，而不是失败样本。"],
    ]
    story.append(info_table(writing_rows, styles, [36 * mm, 128 * mm]))
    story.append(Spacer(1, 8))

    story.append(paragraph("6.1 中间实验态与最终投稿态的区别", styles["h2"]))
    add_bullets(
        story,
        styles,
        [
            "`workspace/code/output/` 里的 tex 更像实验推进期的工作底稿，很多表和口径仍在演化。",
            "`workspace/paper/` 才是投稿包，里面的 paper_main.tex 承担最终叙事职责，很多描述已经不是简单复述 JSON，而是经过解释和收束。",
            "因此，任何关于“论文最终怎么讲”的判断，都必须以 workspace/paper 为准；任何关于“实验过程中发生过什么”的判断，则要回到 workspace/code/output 与状态文件。",
        ],
    )

    story.append(paragraph("6.2 写作真正做了哪些“补实验而非润色”", styles["h2"]))
    add_bullets(
        story,
        styles,
        [
            "confusion matrix 不是锦上添花，而是暴露旧 validator 致命安全洞的危机实验。",
            "Pass 8 feasibility certifier 不是形式扩展，而是把“不安全接受不可行意图”从 72% 压到 0% 的决定性修复。",
            "independent oracle 用来回答“是不是你自己的 certifier 在自证正确”。",
            "runtime benchmark 用来回答“这样一层验证器是不是太慢，不能用于真实系统”。",
            "reachability separation 用来回答“为什么极区场景看起来 GNN 比 Dijkstra 差”。",
            "cross-constellation 与 polar exclusion 用来主动下调 GNN 的泛化 claim，避免过度吹大。",
        ],
    )

    story.append(paragraph("6.3 当前稿件最成熟的叙事姿态", styles["h2"]))
    add_paragraphs(
        story,
        styles,
        """
        当前最成熟、也最稳的写法，是把这篇文章当作“一个被验证的意图驱动约束路由系统”来读。它的贡献顺序应该是：ConstraintProgram 作为桥梁，LLM compiler 负责语义映射，deterministic validator 负责安全兜底，GNN/Dijkstra 负责执行层路由。

        如果把论文理解成“学习型路由算法论文”，你会觉得 GNN 在 OOD 星座和极区强约束上不够强；如果把它理解成“LLM 网络自动化论文”，你又会觉得高 abstain rate 和手工 fallback 太保守。只有把它放回“高风险系统中的语义接口层 + 安全执行层”这个定位里，最终稿的所有结构安排才最合理。
        """,
    )
def build_audit_section(story: list, styles: dict[str, ParagraphStyle], ctx: dict) -> None:
    full = ctx["ablation_full"]
    benchmark = ctx["benchmark_eval"]
    rule = ctx["ablation_rule_based"]
    model_4b = ctx["model_4b"]
    ood = ctx["ood_eval"]
    confusion = ctx["confusion"]
    reach = ctx["reachability"]
    cross = ctx["cross_constellation"]
    paper_log = ctx["paper_log"]

    story.append(paragraph("7. 全面审视：哪些结论最强，哪些地方必须谨慎", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        """
        这一节不是替论文做宣传，而是站在你要“全面审视这篇文章”的角度，把最强结论、边界条件、口径漂移和潜在审稿风险全部摊开。对这个项目来说，真正成熟的地方在于：它有不少值得信的系统结果；真正需要警惕的地方在于：这些结果不是由一套完全统一的实验脚本直接无缝产出的。
        """,
    )

    story.append(paragraph("7.1 目前最强、最值得信的结论", styles["h2"]))
    strong_rows = [
        ["最强结论 1", "GNN cost-to-go 路由在训练星座上已经稳定匹配 Dijkstra：99.8% PDR、17x speedup。这是 Phase A 最扎实的工程资产。"],
        ["最强结论 2", f"LLM compiler 在 few-shot + verifier + repair loop 条件下，显著优于 zero-shot 和 rule-based parser；当前中间态 full pipeline 达到 {format_pct(full['compiled_rate'])} compiled、{format_pct(full['full_match_rate'])} full match。"],
        ["最强结论 3", f"Pass 8 feasibility certifier 显著提高安全性：3-way confusion matrix 中 infeasible unsafe acceptance={format_pct(confusion['safety']['infeasible_unsafe_accept_rate'])}，运行时中位数仅 {format_ms(ctx['runtime']['all']['median_ms'])}。"],
        ["最强结论 4", "Reachability separation 有效解释了 polar/compositional 场景 raw PDR 差距：当 reachability 只有约 24% 时，raw gap 不应直接读成路由质量退化。"],
        ["最强结论 5", "论文最终稿对 GNN 泛化的 claim 是克制的：海拔变化可泛化，倾角变化不行；GNN 更像 optional accelerator，而不是普适路由器。"],
    ]
    story.append(info_table(strong_rows, styles, [36 * mm, 128 * mm]))
    story.append(Spacer(1, 8))

    story.append(paragraph("7.2 最关键的风险：三套指标口径并存", styles["h2"]))
    add_paragraphs(
        story,
        styles,
        """
        当前仓库里至少同时存在三套实验口径。第一套是 ablation 口径：随机边时延、240 条意图全量分母，核心结果是 97.9 / 91.7 / 86.2。第二套是 benchmark rerun 口径：距离型边时延、同样 240 条全量分母，当前 `benchmark_eval_240.json` 的结果是 85.0 / 73.3 / 70.4。第三套是论文主文口径：把 rerun 中的分子改除以“193 feasible / 47 infeasible”的人为定义分母，得到 98.4 / 87.6，并混入 confusion、rule-based、4B 等其他来源的数字。

        这意味着最严重的问题不是“某个数字抄错了”，而是不同表格与段落混用了不同几何假设、不同分母定义，以及 Pass 8 之后人工整理过的 feasible 集合。换句话说，论文是有证据基础的，但不是由一套单一、可一键重放的流水线直接吐出来的。
        """,
    )

    drift_rows = [
        ["口径 A：ablation_full.json", f"随机边时延；240 全量分母；compiled={format_pct(full['compiled_rate'])}；full match={format_pct(full['full_match_rate'])}；avg latency={format_seconds_from_ms(full['avg_latency_ms'])}。"],
        ["口径 B：benchmark_eval_240.json", f"距离型边时延；240 全量分母；compiled={format_pct(benchmark['compiled_rate'])}；full match={format_pct(benchmark['full_match_rate'])}；avg latency={format_seconds_from_ms(benchmark['avg_latency_ms'])}。"],
        ["口径 C：paper_main.tex", "193 feasible / 47 infeasible 分母；摘要与正文使用 compiled=98.4%、full semantic match=87.6%。这组数值不能从当前命名 JSON 直接一键复现。"],
        ["直接后果", "同一个“full match”在不同表里可能并不是同一件事：有的以 240 为分母，有的以 193 为分母，有的还把安全拒绝率写进类别表。"],
    ]
    story.append(info_table(drift_rows, styles, [40 * mm, 124 * mm]))
    story.append(Spacer(1, 8))

    story.append(paragraph("7.3 指标漂移具体体现在哪里", styles["h2"]))
    add_bullets(
        story,
        styles,
        [
            "`eval_ablations.py` 与 `eval_benchmark.py` 的边时延建模不同。前者用随机时延，后者用 haversine 距离除以传播速度，这足以改变大量 deadline/feasibility 判定。",
            "当前 `benchmark_eval.log` 与 `benchmark_rerun_log.txt` 不是同一轮结果，后者才与 `benchmark_eval_240.json` 一致，说明旧日志不应继续当主结果使用。",
            "论文中的 98.4 / 87.6 依赖一个没有独立落盘的 17 条“Pass 8 新发现 infeasible”子集，因此读者无法只靠当前 JSON 文件完全复算。",
            "`paper_new_tables.tex` 中的 `Infeasible 73.3%` 实际对应 confusion matrix 的 22/30 reject，而不是 compiler full match。这是 safety 指标与 accuracy 指标混写。",
            "4B 表格也有新旧版本混杂：旧表仍保留 n=60 的 partial run，而当前 JSON 已是 240 条全量结果。",
            "同一篇 paper_main.tex 中，“77.9% first try”与“98.4% feasible compiled”来自不同分母，因此“repair loop 额外提升 20.5pp”的表述并非严格同口径差值。",
        ],
    )

    story.append(paragraph("7.4 哪些问题是论文的实质性弱点", styles["h2"]))
    weakness_rows = [
        ["高 abstain rate", f"当前 coverage 只有 {confusion['coverage']['coverage_rate']}%，也就是一半以上程序无法被 certifier 明确 accept/reject。这是安全保守策略，但会被质疑实用性。"],
        ["协议复杂度高", "编译、验证、grounding、routing 多层叠加后，系统很强，但审稿人会问：真实运营流程里是否愿意接受这条长链条。"],
        ["GNN 泛化边界", f"在 97° SSO 上 GNN PDR 只有 {cross['sso_97']['gnn_pdr_mean']:.2f}%，而 Dijkstra 是 {cross['sso_97']['dijkstra_pdr_mean']:.2f}%。这会限制“跨星座通用路由器”的说法。"],
        ["综合 OOD 证据仍有限", f"OOD paraphrase 总体 full match={format_pct(ood['full_match_rate'])}，但 compositional OOD 的样本仍小，不能过度推断。"],
        ["写作管理时滞", f"STATE.md 与当前 log 在页数/box 警告上不一致，说明最终投稿准备阶段仍有少量版本管理摩擦。"],
    ]
    story.append(info_table(weakness_rows, styles, [34 * mm, 130 * mm]))
    story.append(Spacer(1, 8))

    story.append(paragraph("7.5 如果你要用一句话评价这篇文章", styles["h2"]))
    add_paragraphs(
        story,
        styles,
        """
        最公允的评价不是“它已经把自然语言卫星网络自动化彻底做成了”，也不是“它只是把几个老模块拼起来”。更准确的说法是：它成功地把一个高风险问题分解成了多个可验证的低风险模块，并且在路由、编译、安全三层都拿出了有说服力的系统证据；但它的论文主结果存在明显的协议漂移和手工口径整合，因此读者必须区分“系统方向是成立的”与“最终数字表述已经完全自动可复现”这两件事。
        """,
    )
def build_file_guide_section(story: list, styles: dict[str, ParagraphStyle], ctx: dict) -> None:
    reach = ctx["reachability"]
    story.append(paragraph("8. 回到仓库时，你应该按什么顺序读", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        """
        如果你希望在读完本报告后再回到项目里自查，我建议按“先抓大脉络，再看中间证据，最后看最终稿”的顺序。这样最不容易被新旧文件混杂搞晕。
        """,
    )

    reading_rows = [
        ["第一站", "project-brief.md。看项目为什么立成这样，尤其是从父项目继承了哪些失败经验和 research red lines。"],
        ["第二站", "experiment-memory.md。看 16 个 iteration 的主线、每一步 status、drift decision 和 branch outcome。"],
        ["第三站", "STATE.md。看项目后期自述、Phase A/Phase B 汇总，以及写作结束时作者自己认为最重要的结论。"],
        ["第四站", "review-state.json 与 workspace/dashboard-data.json。用来恢复 4 月 7 日和 4 月 8 日上午两个冻结时点。"],
        ["第五站", "workspace/code/output/*.json。重点看 ablation_full、benchmark_eval_240、verifier_confusion_matrix、reachability_separation、cross_constellation_gnn、pass8_runtime。"],
        ["第六站", "workspace/code/output/paper_main.tex。它是 draft / 中间稿窗口，便于观察论文怎样从实验笔记过渡到完整叙事。"],
        ["第七站", "workspace/paper/paper_main.tex 与 paper_new_tables.tex。这里是最终投稿态。所有关于“论文最终怎么写”的判断都应以这里为准。"],
        ["最后一站", "workspace/paper/paper_main.log。用来核对当前编译现实，而不是只相信 STATE.md 的自述。"],
    ]
    story.append(info_table(reading_rows, styles, [30 * mm, 134 * mm]))
    story.append(Spacer(1, 8))

    story.append(paragraph("9. 给你的最终学习建议", styles["h1"]))
    add_bullets(
        story,
        styles,
        [
            "先把这篇文章理解成一篇“高风险系统如何分层设计”的论文，再去读它的技术细节。",
            "读 GNN 时，重点盯住“为什么学 cost-to-go”；读 LLM 时，重点盯住“为什么只做编译不做控制”；读 validator 时，重点盯住“为什么安全边界必须是确定性的”。",
            f"读 polar/compositional 实验时，记得先看 reachability fraction 只有约 {format_pct(reach['polar_avoidance']['reachability_fraction'])} 这一事实，再去看 raw PDR。",
            "读表格时一定问三个问题：分母是谁、边时延如何定义、这是 accuracy 指标还是 safety 指标。",
            "如果你未来还想继续推进这个方向，最优先的工作不是继续堆新模型，而是先把 evaluation protocol 统一，把 feasible-set 变化自动落盘，保证论文主结果可以由单一流水线复现。",
        ],
    )
    add_paragraphs(
        story,
        styles,
        """
        从项目管理的角度，这篇文章还教会你另一件很重要的事：AI 主导科研并不等于“让 AI 自己乱试”，而是要不断把失败经验固化为边界条件，把模型自由度压缩到它真正有价值的地方。本项目之所以比父项目顺，就是因为它做到了这一点。
        """,
    )


def build_report() -> Path:
    register_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    ctx = gather_context()

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="leo-intent-routing 项目演化与论文全景报告",
        author="OpenAI Codex",
    )

    story: list = []
    build_cover(story, styles, ctx)
    build_summary_section(story, styles, ctx)
    build_evidence_section(story, styles, ctx)
    build_timeline_section(story, styles, ctx)
    build_teaching_section(story, styles, ctx)
    build_writing_section(story, styles, ctx)
    build_audit_section(story, styles, ctx)
    build_file_guide_section(story, styles, ctx)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT_PDF


if __name__ == "__main__":
    pdf_path = build_report()
    print(pdf_path)
