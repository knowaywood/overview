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
    high_level_idea: str          # 高层次理念
    architecture: str             # 架构
    components: List[str] = field(default_factory=list)  # 关键组件 / 子模块
    theory: str = ""              # 理论



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


#      Markdown Parsing 部分

_SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "related works": "related_work",
    "background": "related_work",        
    "method": "method",
    "methods": "method",
    "model architecture": "method",     
    "architecture": "method",
    "experiments": "experiments",
    "training": "experiments",            
    "results": "results_analysis",        
    "results and analysis": "results_analysis",
    "analysis": "results_analysis",
    "conclusion": "conclusion_future",    
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
    # 1) 精确匹配
    if lower in _SECTION_ALIASES:
        return _SECTION_ALIASES[lower]

    # 2) 关键词模糊匹配
    if "abstract" in lower:
        return "abstract"
    if "introduction" in lower:
        return "introduction"
    if "related" in lower or "previous work" in lower or "background" in lower:
        return "related_work"
    if "method" in lower or "approach" in lower or "model" in lower or "architecture" in lower:
        return "method"
    if "experiment" in lower or "evaluation" in lower or "setup" in lower:
        return "experiments"
    if "result" in lower or "analysis" in lower:
        return "results_analysis"
    if "conclusion" in lower or "future work" in lower or "discussion" in lower:
        return "conclusion_future"
    if "appendix" in lower or "supplementary" in lower:
        return "appendix"

    # 3) 回退：空格变下划线
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


_BULLET_PATTERN = re.compile(
    r"""^(
        [\-\*\+\u2022]         # -, *, +, 或 Unicode 实心圆点 •
        |
        \d+[\.\)]              # 1.  2)  3.
        |
        [a-zA-Z][\.\)]         # a.  b)  A.  B)
    )\s+(.*)$
    """,
    re.MULTILINE | re.VERBOSE,
)


def _collect_bullets(text: str) -> List[str]:
    items = []
    for m in _BULLET_PATTERN.finditer(text):
        item = m.group(2).strip()
        if item:
            items.append(item)
    return items



def _parse_meta(md_text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    # ========= 1. 选 title：在所有一级标题里挑一个 =========
    # 所有一级标题 '# ...'
    titles = [
        m.group(1).strip()
        for m in re.finditer(r"^#\s+(.+?)\s*$", md_text, re.MULTILINE)
    ]

    title: Optional[str] = None
    if titles:
        def score(t: str) -> int:
            s = t.lower()
            # 避免把版权说明当成 title
            if "permission" in s or "attribution" in s:
                return 0
            # 分数越大越优先，这里简单用长度
            return len(t)

        title = max(titles, key=score)
        meta["title"] = title

    lines = md_text.splitlines()

    # ========= 2. authors_raw：标题下面连续的非空且非标题行 =========
    authors_lines: List[str] = []
    if title is not None:
        # 精确找到这一行 '# Attention Is All You Need'
        title_pattern = re.compile(r"^#\s+" + re.escape(title) + r"\s*$")
        title_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if title_pattern.match(line.strip()):
                title_idx = i
                break

        if title_idx is not None:
            for j in range(title_idx + 1, len(lines)):
                line = lines[j].strip()
                # 碰到空行或新的标题就认为作者块结束
                if not line:
                    if authors_lines:
                        break
                    else:
                        continue
                if line.startswith("#"):
                    break
                authors_lines.append(line)

    if authors_lines:
        # 原始作者块保留为 list[str]
        meta["authors_raw"] = authors_lines

        # ====== 可选：顺便解析一个 authors 列表（只有人名） ======
        joined = " ".join(authors_lines)

        # 按 * / † 切分（适配 Transformer 那种写法）
        segments = re.split(r"[\*\†]+", joined)
        authors: List[str] = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            # 带邮箱的基本是联系方式，跳过
            if "@" in seg:
                continue
            # 明显是单位信息的整段也跳过
            if re.search(
                r"\b(google|university|institute|research|brain|college|school|lab|laboratory)\b",
                seg,
                re.IGNORECASE,
            ):
                continue

            seg = re.sub(r"\s+", " ", seg)
            if seg and seg not in authors:
                authors.append(seg)

        if authors:
            meta["authors"] = authors

    # ========= 3. year：只在文首若干行里找，避免误抓 WMT 2014 =========
    head_text = "\n".join(lines[:40])

    # 3.1 显式 Year/Date 字段
    m_year = re.search(
        r"^(Year|Date|年份)\s*:\s*((19|20)\d{2})\b",
        head_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if m_year:
        meta["year"] = int(m_year.group(2))
    else:
        # 3.2 尝试从 arXiv 头部抓年份
        # 例如: arXiv:1706.03762v7 [cs.CL] 12 Dec 2017
        m_arxiv = re.search(
            r"arXiv:\s*\d{4}\.\d{5}v\d+.*?((19|20)\d{2})",
            head_text,
        )
        if m_arxiv:
            meta["year"] = int(m_arxiv.group(1))
        # 如果还是抓不到 year，就不填，避免乱写 2014 之类的

    # ========= 4. preview：保留前 300 个字符用于调试 =========
    meta["preview"] = md_text[:300]

    return meta



# -------- 各 Section 的解析 --------

def _parse_abstract(text: str) -> Abstract:
    """
    抽取摘要：
    - summary：摘要第一段
    - key_contributions：
        1) 优先用 bullet 列表；
        2) 没有 bullet 时，从句子里找包含
           "we propose/present/introduce/show/demonstrate"、
           "our (main) contributions are" 等 pattern 的句子；
        3) 如果还找不到，就退化为摘要前 3 句。
    """
    paragraphs = _split_paragraphs(text)
    summary = paragraphs[0] if paragraphs else ""

    # 1) 先尝试用 bullet
    key_contributions = _collect_bullets(text)

    # 2) 没有 bullet 时，从句子中抽
    if not key_contributions and text.strip():
        # 简单句子切分
        sentences = [
            s.strip()
            for s in re.split(r'(?<=[\.!?。！？])\s+', text.strip())
            if s.strip()
        ]

        # 匹配包含 "we propose..." / "our contributions are..." 等的句子
        pattern = re.compile(
            r"\b("
            r"we\s+(propose|present|introduce|show|demonstrate)|"
            r"our\s+(main\s+)?contributions?\s+are"
            r")\b",
            re.IGNORECASE,
        )
        contrib_sents = [s for s in sentences if pattern.search(s)]

        if contrib_sents:
            key_contributions = contrib_sents
        else:
            # 3) 实在没有明显的贡献句，就取前 3 句兜底
            key_contributions = sentences[:3]

    return Abstract(summary=summary, key_contributions=key_contributions)



def _parse_introduction(text: str) -> Introduction:
    paragraphs = _split_paragraphs(text)
    problem = paragraphs[0] if paragraphs else ""
    motivation = paragraphs[1] if len(paragraphs) > 1 else ""
    contributions = _collect_bullets(text)
    return Introduction(problem=problem, motivation=motivation, contributions=contributions)


def _parse_related_work(text: str) -> RelatedWork:
    """
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

    # 把方法部分的 bullet 列表（包括 “应用场景/子模块/注意力类型”等）统统当作组件
    components = _collect_bullets(text)

    theory = paragraphs[2] if len(paragraphs) > 2 else ""
    return Method(
        high_level_idea=high_level_idea,
        architecture=architecture,
        components=components,
        theory=theory,
    )



def _parse_experiments(text: str) -> Experiments:
    """
    - 先用 bullets 里的关键词判断 datasets / metrics / baselines
    - 如果 bullets 太少，再在自然段里扫一遍
    - 最后一段文字作为 training_details
    """
    bullets = _collect_bullets(text)
    paragraphs = _split_paragraphs(text)

    datasets: List[str] = []
    metrics: List[str] = []
    baselines: List[str] = []

    DATASET_HINTS = [
        "dataset", "data set", "corpus", "benchmark",
        "wmt", "cifar", "imagenet", "mnist", "squad", "ptb",
    ]
    METRIC_HINTS = [
        "bleu", "rouge", "accuracy", "acc", "f1", "precision",
        "recall", "perplexity", "auc", "wer", "cer",
    ]
    BASELINE_HINTS = [
        "baseline", "compare", "compared to", "we follow", "we adopt",
    ]

    def classify_line(line: str):
        l = line.lower()
        if any(k in l for k in DATASET_HINTS):
            datasets.append(line)
        elif any(k in l for k in METRIC_HINTS):
            metrics.append(line)
        elif any(k in l for k in BASELINE_HINTS):
            baselines.append(line)

    # 1) 先用 bullets
    for b in bullets:
        classify_line(b)

    # 2) 如果信息太少，再从自然段里补一补
    if len(datasets) + len(metrics) + len(baselines) < 3:
        for p in paragraphs:
            classify_line(p)

    training_details = paragraphs[-1] if paragraphs else ""

    return Experiments(
        datasets=datasets,
        metrics=metrics,
        baselines=baselines,
        training_details=training_details,
    )


def _parse_results_analysis(text: str) -> ResultsAnalysis:
    paragraphs = _split_paragraphs(text)

    main_results = ""
    ablation = ""
    case_studies = ""

    for p in paragraphs:
        lp = p.lower()
        if not main_results:
            # 含有 "bleu" "accuracy" "state-of-the-art" 等，就当 main results
            if any(k in lp for k in ["bleu", "accuracy", "state-of-the-art", "sota", "outperforms", "improve"]):
                main_results = p
        if ("ablation" in lp or "component-wise" in lp) and not ablation:
            ablation = p
        if ("case study" in lp or "qualitative" in lp or "visualization" in lp) and not case_studies:
            case_studies = p

    # bullets 里查找优点/缺点（你现在这块已经写得不错，可以保留）
    strengths: List[str] = []
    weaknesses: List[str] = []
    for b in _collect_bullets(text):
        lb = b.lower()
        if any(k in lb for k in ["strength", "advantage", "优点", "优势", "benefit"]):
            strengths.append(b)
        elif any(k in lb for k in ["weakness", "limitation", "缺点", "劣势", "drawback"]):
            weaknesses.append(b)

    # 兜底：如果 main_results 仍然为空，就用第一段
    if not main_results and paragraphs:
        main_results = paragraphs[0]

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

    for p in paragraphs:
        lp = p.lower()
        if any(k in lp for k in ["limitation", "drawback", "shortcoming", "局限", "缺点"]):
            limitations.append(p)
        if any(k in lp for k in ["future work", "future direction", "we plan", "we will", "in the future", "未来"]):
            future.append(p)

    # bullets 作为补充
    for b in _collect_bullets(text):
        lb = b.lower()
        if any(k in lb for k in ["limit", "limitation", "缺点", "局限"]):
            limitations.append(b)
        elif any(k in lb for k in ["future", "direction", "work", "未来"]):
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

    相比之前版本，这里对 experiments / results 做了更“宽”的拼接：
    - Experiments: 把 Training 的若干子小节也拼在一起
    - Results: 把机器翻译 & Constituency Parsing 等结果子节也拼在一起
    """
    sections = _split_sections(md_text)
    # print("SECTIONS:", list(sections.keys()))

    # -------- meta --------
    meta = _parse_meta(md_text)

    # -------- abstract & introduction --------
    abstract = _parse_abstract(sections.get("abstract", ""))
    introduction = _parse_introduction(sections.get("introduction", ""))

    # -------- related work --------
    # 先拿典型命名的 section
    related_text_parts: List[str] = []
    for key in ("related_work", "background"):
        if key in sections:
            related_text_parts.append(sections[key])

    # 再扫一遍，顺带把名字里含有 related / background 的也并进去
    for key, text in sections.items():
        if key in ("related_work", "background"):
            continue
        if any(kw in key for kw in ("related", "background")):
            related_text_parts.append(text)

    related_work = _parse_related_work("\n\n".join(t for t in related_text_parts if t.strip()))

    # -------- method --------
    # 把跟模型结构相关的几个 section 拼在一起
    method_text_parts: List[str] = []
    for key in (
        "method",
        "model_architecture",
        "architecture",
        "encoder_and_decoder_stacks",
        "attention",
        "scaled_dot-product_attention",
        "multi-head_attention",
        "position-wise_feed-forward_networks",
        "embeddings_and_softmax",
        "positional_encoding",
        "why_self-attention",
    ):
        if key in sections:
            method_text_parts.append(sections[key])

    method = _parse_method("\n\n".join(t for t in method_text_parts if t.strip()))

    # -------- experiments / training --------
    exp_text_parts: List[str] = []

    # 1) 明确知道的几个 key（像这篇 Transformer）
    for key in (
        "experiments",
        "training",                    # 如果有单独的 Training section
        "training_data_and_batching",
        "hardware_and_schedule",
        "optimizer",
        "regularization",
    ):
        if key in sections:
            exp_text_parts.append(sections[key])

    # 2) 再扫一遍，把名字里带 experiment / training / evaluation / setup 的也拼进来
    for key, text in sections.items():
        if key in (
            "experiments",
            "training",
            "training_data_and_batching",
            "hardware_and_schedule",
            "optimizer",
            "regularization",
        ):
            continue
        if any(kw in key for kw in ("experiment", "experiments", "training", "evaluation", "setup")):
            exp_text_parts.append(text)

    experiments = _parse_experiments("\n\n".join(t for t in exp_text_parts if t.strip()))

    # -------- results & analysis --------
    results_text_parts: List[str] = []

    # 1) 典型 results 命名
    for key in (
        "results_analysis",
        "results",
        "machine_translation",
        "english_constituency_parsing",
    ):
        if key in sections:
            results_text_parts.append(sections[key])

    # 2) 通用：名字里带 result / analysis / ablation / parsing 的子节也拼进去
    for key, text in sections.items():
        if key in (
            "results_analysis",
            "results",
            "machine_translation",
            "english_constituency_parsing",
        ):
            continue
        if any(
            kw in key
            for kw in ("results", "result", "analysis", "ablation", "parsing", "case_study", "case_studies")
        ):
            results_text_parts.append(text)

    results_analysis = _parse_results_analysis(
        "\n\n".join(t for t in results_text_parts if t.strip())
    )

    # -------- conclusion & future work --------
    conclusion_text = (
        sections.get("conclusion_future")
        or sections.get("conclusion", "")
        or sections.get("discussion", "")
        or ""
    )
    conclusion_future = _parse_conclusion_future(conclusion_text)

    # -------- appendix（可选） --------
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
