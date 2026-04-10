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
OUTPUT_PDF = OUTPUT_DIR / "leo-intent-routing-project-evolution-report-2026-04-10.pdf"


def register_fonts() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
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


def find_line(path: Path, pattern: str) -> int:
    regex = re.compile(pattern)
    for idx, line in enumerate(read_text(path).splitlines(), start=1):
        if regex.search(line):
            return idx
    return -1


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


def colored_table(data: list[list[str]], styles: dict[str, ParagraphStyle], widths) -> Table:
    table_data = []
    for row_idx, row in enumerate(data):
        style = styles["body"] if row_idx == 0 else styles["small"]
        table_data.append([paragraph(cell, style) for cell in row])
    table = LongTable(table_data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#102542")),
                ("FONTNAME", (0, 0), (-1, -1), "ReportFont"),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#7d9ab0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d7e5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbfd")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("ReportFont", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(16 * mm, 9 * mm, "leo-intent-routing 二轮严格复审报告")
    canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"第 {canvas.getPageNumber()} 页")
    canvas.restoreState()


def build_context() -> dict:
    state_text = read_text(PROJECT_ROOT / "STATE.md")
    memory_text = read_text(PROJECT_ROOT / "experiment-memory.md")
    review_text = read_text(PROJECT_ROOT / "review-state.json")
    dashboard_text = read_text(PROJECT_ROOT / "workspace" / "dashboard-data.json")
    readme_text = read_text(CODE_ROOT / "README.md")
    claude_text = read_text(CODE_ROOT / "CLAUDE.md")
    project_brief_text = read_text(PROJECT_ROOT / "project-brief.md")

    ablation_full = load_json(OUTPUT_ROOT / "ablation_full.json")
    ablation_rule = load_json(OUTPUT_ROOT / "ablation_rule_based.json")
    ablation_model_4b = load_json(OUTPUT_ROOT / "ablation_model_4b.json")
    benchmark_eval = load_json(OUTPUT_ROOT / "benchmark_eval_240.json")
    ood_eval = load_json(OUTPUT_ROOT / "ood_eval_results.json")
    confusion = load_json(OUTPUT_ROOT / "verifier_confusion_matrix.json")
    pass8_runtime = load_json(OUTPUT_ROOT / "pass8_runtime.json")
    paper_log = parse_paper_log(PAPER_ROOT / "paper_main.log")

    return {
        "state_text": state_text,
        "memory_text": memory_text,
        "review_text": review_text,
        "dashboard_text": dashboard_text,
        "readme_text": readme_text,
        "claude_text": claude_text,
        "project_brief_text": project_brief_text,
        "ablation_full": ablation_full,
        "ablation_rule": ablation_rule,
        "ablation_model_4b": ablation_model_4b,
        "benchmark_eval": benchmark_eval,
        "ood_eval": ood_eval,
        "confusion": confusion,
        "pass8_runtime": pass8_runtime,
        "paper_log": paper_log,
        "refs": {
            "paper_compiled_metric": find_line(PAPER_ROOT / "paper_main.tex", r"\\textbf\{Metrics\.\} \\textit\{Compiled\}: accepted by the full 8-pass"),
            "paper_uniform_protocol": find_line(PAPER_ROOT / "paper_main.tex", r"comparison and ablation tables use uniform"),
            "paper_distance_rerun": find_line(PAPER_ROOT / "paper_main.tex", r"distance-based rerun achieves 98\.4\\% compiled"),
            "paper_rule_based_gap": find_line(PAPER_ROOT / "paper_main.tex", r"40\.0\\% vs\.\\ 99\.0\\%"),
            "paper_ood_fix": find_line(PAPER_ROOT / "paper_main.tex", r"4\.4pp"),
            "paper_safety_733": find_line(PAPER_ROOT / "paper_new_tables.tex", r"73\.3\\% with 7-pass"),
            "paper_compiler_table": find_line(PAPER_ROOT / "paper_new_tables.tex", r"uniform random edge delays"),
            "paper_cross_model_table": find_line(PAPER_ROOT / "paper_new_tables.tex", r"First-try Rate & 42\.9\\% & \\textbf\{93\.3\\%\}"),
            "state_old_main": find_line(PROJECT_ROOT / "STATE.md", r"98\.4% success, 87\.6% full semantic match"),
            "state_old_ablation": find_line(PROJECT_ROOT / "STATE.md", r"97\.9% \| 91\.7% \| 86\.2%"),
            "memory_old_anchor": find_line(PROJECT_ROOT / "experiment-memory.md", r"97\.9% compiled, 86\.2% full match\). 7-pass"),
            "project_brief_7pass": find_line(PROJECT_ROOT / "project-brief.md", r"7-pass verifier"),
            "readme_7pass": find_line(CODE_ROOT / "README.md", r"7-pass deterministic validator"),
            "generate_tables_7pass": find_line(CODE_ROOT / "scripts" / "generate_tables.py", r"7-pass verifier"),
            "generate_tables_chdir": find_line(CODE_ROOT / "scripts" / "generate_tables.py", r"/home/django/leo-gnn-routing"),
            "eval_adv_7pass": find_line(CODE_ROOT / "scripts" / "eval_adversarial_safety.py", r"7-pass deterministic validator"),
            "eval_runtime_7pass": find_line(CODE_ROOT / "scripts" / "eval_pass8_runtime.py", r"7-pass vs 8-pass"),
            "benchmark_log_overall": find_line(OUTPUT_ROOT / "benchmark_eval.log", r"OVERALL\s+240\s+97\.1%"),
            "rerun_log_overall": find_line(OUTPUT_ROOT / "benchmark_rerun_log.txt", r"OVERALL\s+240\s+85\.0%"),
            "review_phase1": find_line(PROJECT_ROOT / "review-state.json", r'"phase": "phase1"'),
            "dashboard_old_action": find_line(PROJECT_ROOT / "workspace" / "dashboard-data.json", r"Paper draft v1 COMPLETE"),
        },
    }


def build_story(ctx: dict) -> list:
    styles = build_styles()
    story: list = []

    full = ctx["ablation_full"]
    rb = ctx["ablation_rule"]
    model4 = ctx["ablation_model_4b"]
    benchmark = ctx["benchmark_eval"]
    ood = ctx["ood_eval"]
    confusion = ctx["confusion"]
    runtime = ctx["pass8_runtime"]
    log_info = ctx["paper_log"]
    refs = ctx["refs"]

    story.append(paragraph("leo-intent-routing 二轮严格复审报告", styles["title"]))
    story.append(paragraph("以 2026-04-09 旧报告为基线，复核改进后的论文口径、证据链与工程治理状态", styles["subtitle"]))
    story.append(paragraph("覆盖范围：Windows 项目层 `C:\\Users\\SchultzDjango\\Desktop\\Files\\Projects\\ClaudeCode` 与 WSL 执行层 `/home/django/leo-gnn-routing`；输出日期：2026-04-10", styles["subtitle"]))
    story.append(Spacer(1, 5 * mm))

    summary_rows = [
        ["本轮总体判断", "最终论文主结果已经明显比 2026-04-09 更自洽，主表基本回到了同一套 uniform-delay / 240-intent 口径；但工程治理层并未同步收口，因此当前状态更接近“论文层修复成功，系统层仍未审计闭环”而不是“项目整体已彻底修好”。"],
        ["最强改进", f"论文现在把 compiler 主结果明确锚定到 uniform random edge delay 主协议，主文与主表使用 {format_pct(full['compiled_rate'])}/{format_pct(full['full_match_rate'])}；Compiled 定义也已改为与代码一致的 full 8-pass acceptance（paper_main.tex:{refs['paper_compiled_metric']}）。"],
        ["最关键残留风险", "distance-based rerun 的 17 条新增 routing-infeasible 样本仍未物化成 manifest；`workspace/paper/` 与 `workspace/code/output/` 仍分叉；README/STATE/experiment-memory/WSL scripts 仍保留旧口径。"],
        ["当前论文编译现实", f"paper_main.log 显示 {log_info['pages']} 页、fatal error={log_info['fatal_errors']}、overfull={log_info['overfull']}、underfull={log_info['underfull']}，当前不是“0 warning 的 clean submission state”。"],
        ["本轮结论等级", "如果只看最终论文：已从“主结果混口径高风险”降到“主结果基本自洽、中等复现风险”。如果看整个项目仓库：仍未达到单一真相源、自动表格生成、全仓同步的投稿工程标准。"],
    ]
    story.append(info_table(summary_rows, styles, [44 * mm, 128 * mm]))
    story.append(Spacer(1, 4 * mm))

    story.append(paragraph("1. 复审目标与方法", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        """
        这份报告不是重新写一遍上次结论，而是把 2026-04-09 报告提出的问题逐条当成审计基线，再检查你这次改进之后，哪些问题被真正修掉，哪些只是换了表述，哪些则仍然滞留在 Windows 管理层、WSL 执行层或自动生成链路里。

        审计方法分为四层：第一层看最终投稿态，也就是 `workspace/paper/paper_main.tex`、`paper_new_tables.tex` 与 `paper_main.log`；第二层看结果工件，也就是 `ablation_full.json`、`ablation_model_4b.json`、`ood_eval_results.json`、`benchmark_eval_240.json`、`benchmark_rerun_log.txt`、`verifier_confusion_matrix.json`；第三层看生成脚本与执行脚本，包括 `generate_tables.py`、`eval_adversarial_safety.py`、`eval_pass8_runtime.py`；第四层看治理文件，包括 `STATE.md`、`experiment-memory.md`、`project-brief.md`、`review-state.json`、`dashboard-data.json`、WSL `README.md` 与 `CLAUDE.md`。

        我同时抽查了 Windows 镜像层与 WSL 主执行层。结论很明确：这次改动主要集中在“最终论文与最终表格”，而不是“全仓库协议治理”。这本身是一种有效修复，但它的作用边界也必须被讲清楚。
        """,
    )

    story.append(paragraph("2. 与 2026-04-09 报告相比，哪些问题已经修了", styles["h1"]))
    fixed_table = [
        ["问题", "2026-04-09 判断", "2026-04-10 复审结果", "当前结论"],
        ["Compiled 定义漂移", "论文把 Compiled 写成 passes 1-7，但代码 success 实际依赖 full verifier", f"paper_main.tex:{refs['paper_compiled_metric']} 已改成“accepted by the full 8-pass validator”；与 compiler.py / verifier.py 一致", "已修复"],
        ["论文主结果混用 98.4/87.6 与 97.9/86.2", "主文和表格跨协议混写", f"paper_main.tex:{refs['paper_uniform_protocol']} 明确把主表锚到 uniform-delay；paper_new_tables.tex:{refs['paper_compiler_table']} 主编译对比表回到 97.9/86.2", "大体修复"],
        ["OOD 4.4pp 降幅基线错误", "4.4pp 实际来自旧基线 86.2", f"paper_main.tex:{refs['paper_ood_fix']} 与 paper_new_tables.tex:85-88 已明确写成“from the uniform-delay template benchmark”", "已修复"],
        ["4B 表旧值污染", "55.4 / 47.1 等值与当前 JSON 不符", f"paper_new_tables.tex:{refs['paper_cross_model_table']} 已改为 57.5 / 42.9 / 54.2 / 59.6，对应 ablation_model_4b.json", "已修复"],
        ["把 infeasible 73.3% 塞进 compiler full-match 表", "accuracy 与 safety 指标混写", "当前 compiler comparison 表已删除 infeasible 那一行，只保留 single / compositional / conditional", "已修复"],
    ]
    story.append(colored_table(fixed_table, styles, [38 * mm, 48 * mm, 62 * mm, 22 * mm]))
    story.append(Spacer(1, 3 * mm))

    story.append(paragraph("3. 当前最可信的主张是什么", styles["h1"]))
    trust_table = [
        ["主张", "当前数值", "主要来源", "审计判断"],
        ["GNN 主路由性能", "99.8% PDR，17x speedup", "eval_results.json、detailed_metrics.json、论文摘要", "可信"],
        ["LLM compiler 主结果（uniform protocol）", f"{format_pct(full['compiled_rate'])} compiled / {format_pct(full['types_match_rate'])} types / {format_pct(full['full_match_rate'])} full", "ablation_full.json、paper_main.tex、paper_new_tables.tex", "可信"],
        ["规则基线与 9B 差距", f"rule-based full={format_pct(rb['full_match_rate'])}；compositional 40.0% vs 99.0%", "ablation_rule_based.json、ablation_full.json、paper_new_tables.tex", "可信"],
        ["4B vs 9B scaling", f"4B compiled={format_pct(model4['compiled_rate'])}；first-try={format_pct(model4['first_try_rate'])}", "ablation_model_4b.json、paper_new_tables.tex", "可信"],
        ["OOD paraphrase", f"{format_pct(ood['compiled_rate'])} compiled，{format_pct(ood['full_match_rate'])} full；相对 uniform 主基线下降 4.4pp", "ood_eval_results.json、ablation_full.json、paper_main.tex", "可信"],
        ["Validator safety", f"infeasible unsafe acceptance={format_pct(confusion['safety']['infeasible_unsafe_accept_rate'])}；coverage={confusion['coverage']['coverage_rate']}%", "verifier_confusion_matrix.json、paper_new_tables.tex、paper_main.tex", "可信"],
        ["Pass 8 runtime", f"median={runtime['all']['median_ms']:.3f} ms，p95={runtime['all']['p95_ms']:.3f} ms", "pass8_runtime.json、paper_new_tables.tex", "可信"],
    ]
    story.append(colored_table(trust_table, styles, [43 * mm, 42 * mm, 67 * mm, 18 * mm]))
    story.append(Spacer(1, 3 * mm))

    story.append(paragraph("4. 严格复审后仍然存在的核心问题", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        f"""
        第一，distance-based rerun 仍然不是单一工件可复现的闭环。当前论文把主结果收回 uniform-delay 口径，这是本轮最关键的正向修复；但 paper_main.tex:{refs['paper_distance_rerun']} 仍然保留了 distance-based rerun 的“85.0/70.4 over all 240”与“98.4/87.6 over 193 feasible”两段补充说明。前者可以由 `benchmark_eval_240.json` 与 `benchmark_rerun_log.txt:{refs['rerun_log_overall']}` 对回，后者仍然缺少 17 条新增 routing-infeasible 样本的显式 manifest。也就是说，这组补充 claim 不再是主表主结论，但它仍然没有进入“可一键追溯”的状态。

        第二，最终投稿层与自动生成层仍然分叉。`workspace/paper/paper_main.tex` 和 `workspace/paper/paper_new_tables.tex` 已按新口径修正，但 `workspace/code/output/paper_main.tex`、`workspace/code/output/paper_new_tables.tex` 仍保留旧版本；`scripts/generate_tables.py:{refs['generate_tables_7pass']}` 仍写 7-pass，且 `scripts/generate_tables.py:{refs['generate_tables_chdir']}` 仍硬编码 `/home/django/leo-gnn-routing`。这说明当前并不存在一个“从 JSON 自动生成最终投稿表格”的单一流水线，最终稿仍然是人工维护层。

        第三，治理层几乎没有跟进本轮论文修复。`STATE.md:{refs['state_old_main']}` 仍把 Phase B headline 写成 98.4/87.6，`STATE.md:{refs['state_old_ablation']}` 又同时保留 97.9/91.7/86.2；`experiment-memory.md:{refs['memory_old_anchor']}` 依然把 97.9/86.2 与 7-pass 写成 anchor claim；`project-brief.md:{refs['project_brief_7pass']}`、WSL/Windows `README.md:{refs['readme_7pass']}` 都仍写 7-pass。`review-state.json:{refs['review_phase1']}` 仍停在 phase1，`dashboard-data.json:{refs['dashboard_old_action']}` 仍停在 draft v1 完成阶段。换句话说，论文层变了，但状态系统没有一起更新。

        第四，仓库中仍未出现 protocol manifest 或 claim manifest。我对 Windows 项目层和 `/home/django/leo-gnn-routing` 进行了文件名级搜索，没有找到任何 `manifest` / `protocol` / `claim` 类型的治理工件。这意味着项目仍旧依赖“人工理解当前哪份结果该信”，而不是“仓库显式声明当前哪份结果是 canonical”。
        """,
    )

    severity_table = [
        ["级别", "问题", "证据", "影响"],
        ["P0", "distance-based 193-feasible 子集仍无 manifest", "paper_main.tex:340-345；benchmark_eval_240.json 无 discovered-infeasible 字段", "补充主张仍非单工件可复算"],
        ["P0", "final paper 与 generated output 分叉", "workspace/paper/* 与 workspace/code/output/* diff 仍显著存在", "自动生成链不能代表最终投稿稿"],
        ["P0", "治理文件未同步", f"STATE.md:{refs['state_old_main']}；experiment-memory.md:{refs['memory_old_anchor']}；README.md:{refs['readme_7pass']}", "外部读者与后续 AI 极易继续引用旧口径"],
        ["P1", "7-pass unsafe acceptance 写成 73.3% 缺乏统一来源", f"paper_new_tables.tex:{refs['paper_safety_733']} 与 STATE.md:115 的 72% 不一致", "安全基线叙述再次出现小型漂移"],
        ["P1", "benchmark_eval.log 与 benchmark_eval_240.json 不一致", f"benchmark_eval.log:{refs['benchmark_log_overall']} vs benchmark_rerun_log.txt:{refs['rerun_log_overall']}", "日志层不可直接作为真相源"],
        ["P1", "WSL 脚本与 README 仍旧口径", f"README.md:{refs['readme_7pass']}；eval_adversarial_safety.py:{refs['eval_adv_7pass']}；eval_pass8_runtime.py:{refs['eval_runtime_7pass']}", "执行层复现实验路径仍带旧叙事"],
        ["P2", "编码与文档清洁度问题仍在", "experiment-memory.md 仍有 mojibake；STATE/README 仍有旧字符污染", "增加人工复核成本"],
        ["P2", "论文排版 warning 未清零", f"paper_main.log: overfull={log_info['overfull']} / underfull={log_info['underfull']}", "不影响主结论，但不是完全 clean 的提交态"],
    ]
    story.append(colored_table(severity_table, styles, [14 * mm, 45 * mm, 67 * mm, 43 * mm]))

    story.append(PageBreak())
    story.append(paragraph("5. 关键证据链逐项解释", styles["h1"]))

    story.append(paragraph("5.1 论文层：主口径已经比上一轮稳定得多", styles["h2"]))
    add_paragraphs(
        story,
        styles,
        """
        本轮最值得肯定的修复，是作者没有再把 98.4/87.6 当作整篇论文的主 headline，而是把 compiler 主表、rule-based 对比表和 cross-model 表全部锚回了 `ablation_full.json` 这一套 uniform-delay / 240-intent 主协议。这个选择让 final paper 至少在“我到底在拿哪一套协议当主结果”这件事上变得清晰了。

        更重要的是，Compiled 指标定义已经与代码语义对齐。`paper_main.tex` 现在明确写的是“accepted by the full 8-pass validator, including Pass 8 feasibility certification”，这与 `intent/compiler.py` 中 `if vr.valid: result.success = True` 的行为一致。相比上一轮，这个修复是实质性的，不是措辞层修饰。

        另外，OOD 的 4.4pp 降幅也被重新锚到了 uniform-delay template benchmark；4B 表中的 `types_match=57.5%` 与 `first_try=42.9%` 也已经回到当前 `ablation_model_4b.json`。这些都说明本轮修复确实针对了上一轮报告指出的具体错位。
        """,
    )

    story.append(paragraph("5.2 结果层：distance-based rerun 仍然半闭合", styles["h2"]))
    result_table = [
        ["工件", "当前数值", "可否单独复现", "复审判断"],
        ["ablation_full.json", "97.9 / 91.7 / 86.2；first-try 93.3", "可以", "现在是论文主 compiler 口径"],
        ["benchmark_rerun_log.txt + benchmark_eval_240.json", "85.0 / 73.3 / 70.4；first-try 77.9", "可以", "distance-based rerun 的 all-240 口径清楚"],
        ["paper_main.tex 中 193-feasible 说明", "98.4 / 87.6 on 193 feasible", "不可以", "仍缺 17 intent manifest 与 canonical summary"],
        ["verifier_confusion_matrix.json", "0/30 unsafe；coverage 46.7%", "可以", "安全工件仍然扎实"],
    ]
    story.append(colored_table(result_table, styles, [43 * mm, 46 * mm, 28 * mm, 52 * mm]))
    add_paragraphs(
        story,
        styles,
        """
        这里最需要严格区分的是：当前问题已经不再是“论文主表把所有协议都搅在一起”，而是“distance-based rerun 被降级为补充证据后，仍没有被完全工程化”。如果这组 98.4/87.6 只是 discussion 中的补充说明，风险已经比上一轮小得多；但如果未来任何摘要、状态页、答辩材料又把它拿回 headline，没有 manifest 的老问题就会重新回来。
        """,
    )

    story.append(paragraph("5.3 生成层：自动化并没有跟上论文修复", styles["h2"]))
    add_bullets(
        story,
        styles,
        [
            f"`scripts/generate_tables.py:{refs['generate_tables_7pass']}` 仍写 7-pass caption，说明表格生成器不是 final paper 的真源。",
            f"`scripts/generate_tables.py:{refs['generate_tables_chdir']}` 仍硬编码 `/home/django/leo-gnn-routing`，说明脚本没有完成 Windows/WSL 双层治理迁移。",
            "`workspace/code/output/paper_main.tex` 与 `workspace/paper/paper_main.tex` 仍显著 diff；同样 `paper_new_tables.tex` 两层也仍 diff。",
            f"`benchmark_eval.log:{refs['benchmark_log_overall']}` 仍打印 97.1 / 91.7 / 31.2 这套旧 summary，而 `benchmark_rerun_log.txt:{refs['rerun_log_overall']}` 与 `benchmark_eval_240.json` 对应的是 85.0 / 73.3 / 70.4。",
        ],
    )
    add_paragraphs(
        story,
        styles,
        """
        这说明当前项目虽然把最终论文修到了一个明显更稳的状态，但“如何从结果自动走到论文”这条链路并没有被修好。只要这一层不收口，后续任何一次重跑、重构、迁移，仍然可能重新制造一轮口径漂移。
        """,
    )

    story.append(paragraph("5.4 治理层：大多数旧状态仍在原地", styles["h2"]))
    governance_table = [
        ["文件", "当前状态", "问题性质"],
        ["STATE.md", "仍把 headline 写成 98.4/87.6，同时保留 97.9/91.7/86.2 表", "管理层双口径并存"],
        ["experiment-memory.md", "仍以 97.9/86.2 + 7-pass 作为 anchor claim，且含 mojibake", "历史锚点未重写"],
        ["project-brief.md", "in_scope 仍写 7-pass verifier", "需求边界未同步"],
        ["review-state.json", "phase 仍是 phase1；warning 仍写 4B incomplete", "阶段快照过期"],
        ["dashboard-data.json", "仍停在 draft v1 / prepare repo 阶段", "仪表盘过期"],
        ["WSL README.md", "仍写 7-pass、97.9/86.2 主结果", "执行层说明过期"],
    ]
    story.append(colored_table(governance_table, styles, [34 * mm, 72 * mm, 63 * mm]))
    add_paragraphs(
        story,
        styles,
        """
        所以，本轮修复最准确的说法不是“项目整体完成整改”，而是“论文主叙事整改有效，但项目系统治理没有同步收敛”。这是一个非常典型、也非常常见的二阶段修复结构：先保住最终稿，再回头治仓库。它不是失败，但必须被诚实命名。
        """,
    )

    story.append(paragraph("6. 对当前论文水平的更新判断", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        """
        如果用 2026-04-09 之前的状态来评价，这篇论文的最大弱点是“主结果口径不干净”；而现在，这个弱点已经显著减轻。当前版本的最终论文，至少已经把 most publishable narrative 收束到了更可辩护的框架：GNN 负责匹配 Dijkstra 并加速，LLM 负责语义编译，8-pass validator 负责安全边界，uniform-delay compiler benchmark 作为主协议，distance-based rerun 作为补充说明而非统领全篇的 headline。

        因此，单看 paper layer，这篇文章已经从“内部口径会被审稿人一眼打穿”的状态，提升到了“主叙事基本成立，但 supplemental evidence chain 仍不够工程化”的状态。它现在更像是一篇已经有系统论文骨架、主表也更稳的稿件，而不是一篇需要从头重写实验口径的稿件。

        但如果把评价对象从“paper pdf”扩展到“整个项目是否进入高质量可复核提交态”，结论仍然要保守。原因很简单：项目治理层、执行脚本层、自动生成层、状态快照层都没有同步完成整改。也就是说，这篇论文现在可以说“比上一轮稳得多”，但还不能说“整个项目已经达到了单一真相源的可审计状态”。
        """,
    )

    level_table = [
        ["维度", "当前判断", "解释"],
        ["论文主叙事一致性", "较好", "主 compiler 表、cross-model、OOD 与 abstract/intro 现在大体对齐"],
        ["实验结果可追溯性", "中等", "uniform 主协议可追；distance-based feasible subset 仍缺 manifest"],
        ["仓库治理成熟度", "较弱", "状态文件、README、WSL scripts、generated outputs 仍不同步"],
        ["投稿准备完整度", "中等偏上", "paper 可以读、可以编译，但还不是 clean and canonical repo state"],
    ]
    story.append(colored_table(level_table, styles, [36 * mm, 27 * mm, 106 * mm]))

    story.append(paragraph("7. 建议的整改顺序", styles["h1"]))
    p0_items = [
        "P0-1：物化 distance-based rerun 的 canonical summary。至少需要一份逐 intent 的 manifest，明确 17 条新增 routing-infeasible 样本是谁、为什么被排到 193 feasible 之外。",
        "P0-2：收口 final paper 生成链。要么让 `workspace/paper/*.tex` 成为唯一真源并显式声明生成层废弃，要么让 `scripts/generate_tables.py` 真正生成 final tables，而不是继续与投稿稿分叉。",
        "P0-3：同步更新 STATE / experiment-memory / project-brief / README / review-state / dashboard。否则仓库外层仍会继续诱导人引用旧口径。",
        "P0-4：修正 7-pass unsafe acceptance 的 73.3% / 72% 漂移，明确到底引用的是旧 confusion metric 还是 adversarial safety metric，不能再混写。",
    ]
    p1_items = [
        "P1-1：清理 `benchmark_eval.log` 这类已与当前 JSON 不一致的日志，或给它们加上 `stale` 标记。",
        "P1-2：在 Windows 与 WSL 根目录下都补 `protocol.md` / `claim_manifest.json` 之类最小治理工件，让未来所有摘要数字都有出处。",
        "P1-3：统一 WSL README、CLAUDE.md、脚本 docstring，把 7-pass 与旧 headline 系统性替换掉。",
    ]
    p2_items = [
        "P2-1：清理 mojibake 与旧字符污染，降低后续人工复核和 AI 接力时的解释噪声。",
        "P2-2：把 paper_main.log 的 box warning 进一步压低，虽然它不影响主结论，但会让投稿态更干净。",
        "P2-3：把旧报告脚本与本次新报告脚本一并纳入治理，让后续项目复盘报告也能明确区分“paper layer 修复”和“repo layer 修复”。",
    ]
    story.append(paragraph("P0：提交前必须处理", styles["h2"]))
    add_bullets(story, styles, p0_items)
    story.append(paragraph("P1：本轮之后尽快补齐", styles["h2"]))
    add_bullets(story, styles, p1_items)
    story.append(paragraph("P2：提升长期维护质量", styles["h2"]))
    add_bullets(story, styles, p2_items)

    story.append(paragraph("8. 结论", styles["h1"]))
    add_paragraphs(
        story,
        styles,
        f"""
        这次改进最重要的成果，是把最终论文从“核心主结果混协议、混分母、混表意”拉回到了“主表口径基本统一”的状态。具体说，当前 paper layer 最可信的 compiler 主结果已经回到了 `ablation_full.json` 这条 uniform-delay 主协议：{format_pct(full['compiled_rate'])} compiled、{format_pct(full['full_match_rate'])} full match、93.3% first-try、compositional 99.0%。相较上一轮，这是真正的结构性改善。

        但同样需要非常明确地说：这次整改并没有把整个工程系统一起修好。`STATE.md`、`experiment-memory.md`、`project-brief.md`、`review-state.json`、`dashboard-data.json`、WSL README、`generate_tables.py`、`eval_adversarial_safety.py`、`eval_pass8_runtime.py` 仍然保留旧叙事或旧协议；`workspace/paper/` 与 `workspace/code/output/` 仍然分叉；distance-based feasible subset 仍缺 manifest。因此当前最准确的结论是：论文层显著变稳，系统层仍需继续治理。

        如果你的目标是“判断这篇论文现在值不值得继续推”，答案比 2026-04-09 更积极；如果你的目标是“判断这个项目现在是否已经达到高质量、单一真相源、全仓自动可复现的提交态”，答案仍然是否。后续工作的重心，不再是继续救论文主表，而是把证据链和治理链真正收口。
        """,
    )

    story.append(paragraph("9. 关键文件索引", styles["h1"]))
    key_files = [
        "`workspace/paper/paper_main.tex`：当前最终投稿主文。",
        "`workspace/paper/paper_new_tables.tex`：当前最终投稿主表。",
        "`workspace/paper/paper_main.log`：当前编译现实（9 页，1 overfull，5 underfull）。",
        "`workspace/code/output/ablation_full.json`：uniform-delay compiler 主协议。",
        "`workspace/code/output/ablation_model_4b.json`：4B cross-model 现行工件。",
        "`workspace/code/output/ood_eval_results.json`：OOD paraphrase 工件。",
        "`workspace/code/output/benchmark_eval_240.json` + `benchmark_rerun_log.txt`：distance-based rerun all-240 工件。",
        "`workspace/code/output/verifier_confusion_matrix.json`：三分类安全工件。",
        "`STATE.md` / `experiment-memory.md` / `project-brief.md` / `review-state.json` / `workspace/dashboard-data.json`：当前仍未同步的治理层文件。",
        "`/home/django/leo-gnn-routing/README.md`、`scripts/generate_tables.py`、`scripts/eval_adversarial_safety.py`、`scripts/eval_pass8_runtime.py`：WSL 侧仍保留旧叙事的关键执行层文件。",
    ]
    add_bullets(story, styles, key_files)

    return story


def main() -> None:
    register_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = build_context()

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="leo-intent-routing 二轮严格复审报告",
        author="OpenAI Codex",
    )
    story = build_story(ctx)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Saved report to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
