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
