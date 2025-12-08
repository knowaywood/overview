import re
from dataclasses import asdict
from pathlib import Path
import json

"""process papaer content into structured data."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# 1. 定义子结构 (Related Work Area)
@dataclass
class RelatedWorkArea:
    area: str
    representative_works: List[str] = field(default_factory=list)
    limitations: str = ""


# 2. 定义各个主要板块
@dataclass
class Abstract:
    summary: str
    key_contributions: List[str] = field(default_factory=list)  # 主要贡献


@dataclass
class Introduction:
    problem: str
    motivation: str  # 动机
    contributions: List[str] = field(default_factory=list)


@dataclass
class RelatedWork:
    areas: List[RelatedWorkArea] = field(default_factory=list)  # 领域列表


@dataclass
class Method:
    high_level_idea: str  # 高层次理念
    architecture: str  # 架构
    loss_functions: List[str] = field(default_factory=list)  # 损失函数
    theory: str = ""  # 理论


@dataclass
class Experiments:
    datasets: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)  # 指标
    baselines: List[str] = field(default_factory=list)  # 基准值
    training_details: str = ""  # 训练细节


@dataclass
class ResultsAnalysis:
    main_results: str
    ablation: str  # 消融实验
    case_studies: str  # 案例研究
    strengths: List[str] = field(default_factory=list)  # 优势分析
    weaknesses: List[str] = field(default_factory=list)  # 劣势分析


@dataclass
class ConclusionFuture:
    summary: str
    limitations: List[str] = field(default_factory=list)  # 局限性
    future_directions: List[str] = field(default_factory=list)  # 未来方向


@dataclass
class Appendix:
    extra_experiments: str = ""
    derivations: str = ""


# 3. 定义主类 (Paper)
@dataclass
class PaperAnalysis:
    meta: Dict[str, Any]  # 存储标题、作者等元数据
    abstract: Abstract
    introduction: Introduction
    related_work: RelatedWork
    method: Method
    experiments: Experiments
    results_analysis: ResultsAnalysis
    conclusion_future: ConclusionFuture
    appendix: Optional[Appendix] = None

# ==============================
#      Markdown Parsing 部分
# ==============================

_SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "related works": "related_work",
    "background": "related_work",         # 像这篇 Transformer 论文就用 Background
    "method": "method",
    "methods": "method",
    "model architecture": "method",       # 3 Model Architecture -> method
    "architecture": "method",
    "experiments": "experiments",
    "training": "experiments",            # 5 Training -> experiments
    "results": "results_analysis",        # 6 Results -> results_analysis
    "results and analysis": "results_analysis",
    "analysis": "results_analysis",
    "conclusion": "conclusion_future",    # 7 Conclusion -> conclusion_future
    "conclusion and future work": "conclusion_future",
    "future work": "conclusion_future",
    "appendix": "appendix",
}


def _normalize_heading(heading: str) -> str:
    """
    标题归一化：
    - 去掉前导编号：'3.1 Encoder and Decoder Stacks' -> 'encoder and decoder stacks'
    - 映射到统一 section 名称（_SECTION_ALIASES）
    - 其他的用小写 + 下划线
    """
    key = heading.strip()
    key = re.sub(r"\s+", " ", key)
    # 去掉开头的 1 / 1. / 3.4. 这种编号
    key = re.sub(r"^[0-9]+(\.[0-9]+)*\s+", "", key)
    lower = key.lower()
    if lower in _SECTION_ALIASES:
        return _SECTION_ALIASES[lower]
    return lower.replace(" ", "_")


def _split_sections(md_text: str) -> Dict[str, str]:
    """
    按 Markdown 标题切分成若干 section.
    返回 dict: {section_name: content}
    其中可能包含一个特殊 key: "_preamble" 表示第一个标题之前的内容。
    """
    sections: Dict[str, str] = {}
    matches = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", md_text, re.MULTILINE))

    if not matches:
        sections["_preamble"] = md_text.strip()
        return sections

    # preamble
    first = matches[0]
    if first.start() > 0:
        sections["_preamble"] = md_text[: first.start()].strip()

    # each section
    for i, m in enumerate(matches):
        heading_text = m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[start:end].strip()
        key = _normalize_heading(heading_text)
        # 如果同名 section 出现多次，就拼接在一起
        if key in sections:
            sections[key] += "\n\n" + body
        else:
            sections[key] = body

    return sections


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _collect_bullets(text: str) -> List[str]:
    return [
        m.group(1).strip()
        for m in re.finditer(r"^[\-\*\+]\s+(.*)$", text, re.MULTILINE)
        if m.group(1).strip()
    ]


def _parse_meta(md_text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    # 所有一级标题 '# ...'
    titles = [
        m.group(1).strip()
        for m in re.finditer(r"^#\s+(.+?)\s*$", md_text, re.MULTILINE)
    ]
    if titles:
        def score(t: str) -> int:
            s = t.lower()
            # 避免把版权说明当成 title
            if "permission" in s or "attribution" in s:
                return 0
            return len(t)
        meta["title"] = max(titles, key=score)

    # Authors: 对这篇 md 不一定好抓，这里只做一个非常粗的 heuristic：
    # 找含有多个 email 的行
    author_lines = []
    for line in md_text.splitlines():
        if "@" in line and len(re.findall(r"[\w\.-]+@[\w\.-]+", line)) >= 2:
            author_lines.append(line.strip())
    if author_lines:
        meta["authors_raw"] = author_lines

    # Year / Date （简单抓 4 位数字年）
    m_year = re.search(
        r"(19|20)\d{2}", md_text
    )
    if m_year:
        meta["year"] = int(m_year.group(0))

    meta["preview"] = md_text[:300]

    return meta


# -------- 各 Section 的解析 --------

def _parse_abstract(text: str) -> Abstract:
    paragraphs = _split_paragraphs(text)
    summary = paragraphs[0] if paragraphs else ""
    key_contributions = _collect_bullets(text)
    return Abstract(summary=summary, key_contributions=key_contributions)


def _parse_introduction(text: str) -> Introduction:
    paragraphs = _split_paragraphs(text)
    problem = paragraphs[0] if paragraphs else ""
    motivation = paragraphs[1] if len(paragraphs) > 1 else ""
    contributions = _collect_bullets(text)
    return Introduction(problem=problem, motivation=motivation, contributions=contributions)


def _parse_related_work(text: str) -> RelatedWork:
    """
    对于这篇 Transformer 论文：
    - 我们会把 'Background' 也一起并入 related_work 中
    - 如果没有三级标题，就把整个 section 当成一个 area='overall'
    """
    areas: List[RelatedWorkArea] = []
    if not text.strip():
        return RelatedWork(areas=[])

    sub_matches = list(re.finditer(r"^(#{3,6})\s+(.+?)\s*$", text, re.MULTILINE))
    if not sub_matches:
        reps = _collect_bullets(text)
        paras = _split_paragraphs(text)
        limitations = paras[-1] if len(paras) > 1 else ""
        areas.append(
            RelatedWorkArea(
                area="overall",
                representative_works=reps,
                limitations=limitations,
            )
        )
        return RelatedWork(areas=areas)

    for i, m in enumerate(sub_matches):
        area_name = m.group(2).strip()
        start = m.end()
        end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(text)
        body = text[start:end].strip()

        reps = _collect_bullets(body)
        paras = _split_paragraphs(body)
        limitations = paras[-1] if len(paras) > 1 else ""
        areas.append(
            RelatedWorkArea(
                area=area_name,
                representative_works=reps,
                limitations=limitations,
            )
        )

    return RelatedWork(areas=areas)


def _parse_method(text: str) -> Method:
    paragraphs = _split_paragraphs(text)
    high_level_idea = paragraphs[0] if paragraphs else ""
    architecture = paragraphs[1] if len(paragraphs) > 1 else ""
    loss_functions = _collect_bullets(text)
    theory = paragraphs[2] if len(paragraphs) > 2 else ""
    return Method(
        high_level_idea=high_level_idea,
        architecture=architecture,
        loss_functions=loss_functions,
        theory=theory,
    )


def _parse_experiments(text: str) -> Experiments:
    """
    对于这篇论文：
    - 5 Training + 6 Results 里会出现各种表格/指标，这里做一个非常粗的 heuristic：
      - bullet 中含 'dataset'/'data' -> datasets
      - bullet 中含 'bleu'/'acc'/'f1' -> metrics
      - 其他 bullet -> baselines
    - 最后一段文字当成 training_details
    """
    bullets = _collect_bullets(text)
    datasets: List[str] = []
    metrics: List[str] = []
    baselines: List[str] = []

    for b in bullets:
        lb = b.lower()
        if "dataset" in lb or "data" in lb:
            datasets.append(b)
        elif any(k in lb for k in ["bleu", "accuracy", "acc", "f1", "rouge"]):
            metrics.append(b)
        else:
            baselines.append(b)

    paragraphs = _split_paragraphs(text)
    training_details = paragraphs[-1] if paragraphs else ""

    return Experiments(
        datasets=datasets,
        metrics=metrics,
        baselines=baselines,
        training_details=training_details,
    )


def _parse_results_analysis(text: str) -> ResultsAnalysis:
    paragraphs = _split_paragraphs(text)
    main_results = paragraphs[0] if paragraphs else ""
    ablation = ""
    case_studies = ""

    for p in paragraphs[1:]:
        lp = p.lower()
        if "ablation" in lp or "消融" in lp:
            ablation = p
        if "case" in lp or "案例" in lp:
            case_studies = p

    strengths: List[str] = []
    weaknesses: List[str] = []
    for b in _collect_bullets(text):
        lb = b.lower()
        if any(k in lb for k in ["strength", "advantage", "优点", "优势"]):
            strengths.append(b)
        elif any(k in lb for k in ["weakness", "limitation", "缺点", "劣势"]):
            weaknesses.append(b)

    return ResultsAnalysis(
        main_results=main_results,
        ablation=ablation,
        case_studies=case_studies,
        strengths=strengths,
        weaknesses=weaknesses,
    )


def _parse_conclusion_future(text: str) -> ConclusionFuture:
    paragraphs = _split_paragraphs(text)
    summary = paragraphs[0] if paragraphs else ""
    limitations: List[str] = []
    future: List[str] = []

    for b in _collect_bullets(text):
        lb = b.lower()
        if any(k in lb for k in ["limit", "limitation", "缺点", "局限"]):
            limitations.append(b)
        if any(k in lb for k in ["future", "direction", "work", "未来"]):
            future.append(b)

    return ConclusionFuture(
        summary=summary,
        limitations=limitations,
        future_directions=future,
    )


def _parse_appendix(text: str) -> Appendix:
    paragraphs = _split_paragraphs(text)
    extra_experiments = paragraphs[0] if paragraphs else ""
    derivations = paragraphs[1] if len(paragraphs) > 1 else ""
    return Appendix(extra_experiments=extra_experiments, derivations=derivations)


# -------- 对外主入口函数 --------

def parse_paper_markdown(md_text: str) -> PaperAnalysis:
    """
    主入口：给一段整篇论文的 Markdown 字符串，返回 PaperAnalysis 对象。
    对 Transformer 这篇 md：
    - Abstract -> abstract
    - 1 Introduction -> introduction
    - 2 Background -> related_work
    - 3 Model Architecture (+ 子节) -> method
    - 5 Training -> experiments
    - 6 Results -> results_analysis
    - 7 Conclusion -> conclusion_future
    """
    sections = _split_sections(md_text)
    meta = _parse_meta(md_text)

    abstract = _parse_abstract(sections.get("abstract", ""))
    introduction = _parse_introduction(sections.get("introduction", ""))

    # related_work: 可能有 'related_work', 'background'
    related_text_parts = [
        sections.get("related_work", ""),
        sections.get("background", ""),
    ]
    related_work = _parse_related_work("\n\n".join(t for t in related_text_parts if t))

    # method: 'method', 'model_architecture', 'architecture'
    method_text_parts = [
        sections.get("method", ""),
        sections.get("model_architecture", ""),
        sections.get("architecture", ""),
    ]
    method = _parse_method("\n\n".join(t for t in method_text_parts if t))

    # experiments: 'experiments', 'training'
    exp_text_parts = [
        sections.get("experiments", ""),
        sections.get("training", ""),
    ]
    experiments = _parse_experiments("\n\n".join(t for t in exp_text_parts if t))

    # results_analysis: 'results_analysis', 'results'
    results_text = sections.get("results_analysis") or sections.get("results", "") or ""
    results_analysis = _parse_results_analysis(results_text)

    conclusion_text = (
        sections.get("conclusion_future")
        or sections.get("conclusion", "")
        or ""
    )
    conclusion_future = _parse_conclusion_future(conclusion_text)

    appendix_section = sections.get("appendix")
    appendix = _parse_appendix(appendix_section) if appendix_section is not None else None

    return PaperAnalysis(
        meta=meta,
        abstract=abstract,
        introduction=introduction,
        related_work=related_work,
        method=method,
        experiments=experiments,
        results_analysis=results_analysis,
        conclusion_future=conclusion_future,
        appendix=appendix,
    )


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Parse markdown paper into PaperAnalysis dataclass."
    )
    # 允许 md_path 可选，并给一个默认值
    parser.add_argument(
        "md_path",
        type=str,
        nargs="?",  # <- 让这个位置参数变成“0 或 1 个”
        default="examples/Example/md/1706.03762v7.md",
        help="Path to markdown file of the paper (default: %(default)s).",
    )
    args = parser.parse_args()

    md_path = Path(args.md_path)
    md_text = md_path.read_text(encoding="utf-8")

    paper = parse_paper_markdown(md_text)
    print(json.dumps(asdict(paper), ensure_ascii=False, indent=2))
