"""Meta Data extraction from pdf."""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import fitz
import requests


@dataclass
class PaperMetadata:
    """Meta structure of paper metadata."""

    title: Optional[str]
    publication_year: Optional[int]
    cited_by_count: Optional[int]
    institution_lead: str
    openalex_url: Optional[str]
    pdf_url: Optional[str]
    authors: List[str]
    concepts: List[str]
    citation_trend_last_3y: Dict[int, int]
    abstract_full: Optional[str]


class PaperMetaAnalyzer:
    """PDF论文分析器"""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    DOI_PATTERN = r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)"
    ARXIV_PATTERN = r"arXiv:(\d{4}\.\d{4,5})"

    def __init__(self, file_paths: list[str]) -> None:
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

        meta_data = []
        for file_path in file_paths:
            meta_data.append(self.get_paper_metadata(file_path))
            print(f"DONE : {file_path}\n")
        self.metadata = meta_data

    @staticmethod
    def reconstruct_abstract(
        inverted_index: Optional[Dict[str, List[int]]],
    ) -> Optional[str]:
        """将OpenAlex的倒排索引摘要还原为可读文本"""
        if not inverted_index:
            return None

        # 展平为 (位置, 单词) 对并排序
        word_positions = [
            (pos, word)
            for word, positions in inverted_index.items()
            for pos in positions
        ]
        word_positions.sort(key=lambda x: x[0])

        return " ".join(word for _, word in word_positions)

    def extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        """Exatract text from the first page of pdf."""
        try:
            with fitz.open(file_path) as doc:
                if len(doc) < 1:
                    print("❌ PDF 内容为空")
                    return None
                text = doc[0].get_text()
                return str(text) if text else None
        except Exception as e:
            print(f"❌ PDF读取失败: {e}")
            return None

    def extract_ids_from_text(self, text: str) -> Dict[str, Optional[str]]:
        """从文本和文件名中提取DOI和arXiv ID"""
        ids: Dict[str, Optional[str]] = {"doi": None, "arxiv": None}

        # 提取DOI
        doi_match = re.search(self.DOI_PATTERN, text, re.IGNORECASE)
        if doi_match:
            ids["doi"] = doi_match.group(1)
            print(f"✅ 提取到 DOI: {ids['doi']}")
            return ids

        arxiv_match = re.search(self.ARXIV_PATTERN, text, re.IGNORECASE)

        if arxiv_match:
            ids["arxiv"] = arxiv_match.group(1)
            return ids

        print("❌ 未能识别 DOI 或 arXiv ID。请检查 PDF 是否为扫描件。")
        return ids

    def get_paper_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """主入口：传入PDF路径，返回详细的元数据字典"""
        if not os.path.exists(file_path):
            print(f"❌ 错误: 文件不存在 -> {file_path}")
            return None

        filename = os.path.basename(file_path)
        print(f"📄 正在分析: {filename}")

        # 提取PDF文本
        text = self.extract_text_from_pdf(file_path)
        if not text:
            return None

        # 提取ID
        ids = self.extract_ids_from_text(text)

        # 根据ID类型查询
        doi = ids.get("doi")
        arxiv = ids.get("arxiv")

        if doi:
            return self.query_by_doi(doi)
        elif arxiv:
            clean_id = arxiv.split("v")[0]  # 清洗版本号
            return self.query_by_arxiv(clean_id)

        print("❌ 未能识别 DOI 或 arXiv ID。请检查 PDF 是否为扫描件。")
        return None

    def query_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """通过DOI查询OpenAlex"""
        doi_url = f"https://doi.org/{doi}" if not doi.startswith("http") else doi
        api_url = f"https://api.openalex.org/works/{doi_url}"
        print(f"🔍 [DOI模式] 查询 OpenAlex: {doi}")
        return self._fetch_metadata(api_url, mode="direct")

    def query_by_arxiv(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """通过arXiv ID查询OpenAlex"""
        api_url = f"https://api.openalex.org/works?search={arxiv_id}"
        return self._fetch_metadata(api_url, mode="search")

    def _fetch_metadata(self, url: str, mode: str) -> Optional[Dict[str, Any]]:
        """从OpenAlex API获取元数据"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()
            result = self._extract_result(data, mode)
            if not result:
                return None

            return self._parse_metadata(result)

        except requests.RequestException as e:
            print(f"❌ API请求失败: {e}")
            return None
        except Exception as e:
            print(f"❌ 数据解析错误: {e}")
            return None

    def _extract_result(
        self, data: Dict[str, Any], mode: str
    ) -> Optional[Dict[str, Any]]:
        """从API响应中提取结果"""
        if mode == "search":
            if data["meta"]["count"] == 0:
                print("❌ 搜索未找到匹配记录")
                return None
            return data["results"][0]  # 取置信度最高的第一条
        return data  # 直接模式返回的就是对象

    def _parse_metadata(self, result: Dict[str, Any]) -> PaperMetadata:
        """解析OpenAlex API返回的数据"""
        # 处理摘要
        abstract_text = self.reconstruct_abstract(result.get("abstract_inverted_index"))
        abstract_preview = (
            abstract_text[:300] + "..." if abstract_text else "No Abstract Available"
        )

        # 提取第一作者机构
        institution = self._extract_institution(result)

        # 提取引用趋势
        citation_trend = self._extract_citation_trend(result)

        return PaperMetadata(
            title=result.get("title"),
            publication_year=result.get("publication_year"),
            cited_by_count=result.get("cited_by_count"),
            institution_lead=institution,
            openalex_url=result.get("ids", {}).get("openalex"),
            pdf_url=result.get("open_access", {}).get("oa_url"),
            authors=self._extract_authors(result),
            concepts=self._extract_concepts(result),
            citation_trend_last_3y=citation_trend,
            abstract_full=abstract_text,
        )

    def _extract_institution(self, result: Dict[str, Any]) -> str:
        """提取第一作者机构"""
        if result.get("authorships"):
            first_author = result["authorships"][0]
            if first_author.get("institutions"):
                return first_author["institutions"][0]["display_name"]
        return "Unknown"

    def _extract_authors(self, result: Dict[str, Any]) -> List[str]:
        """提取作者列表（前5位）"""
        return [
            ship["author"]["display_name"] for ship in result.get("authorships", [])[:5]
        ]

    def _extract_concepts(self, result: Dict[str, Any]) -> List[str]:
        """提取概念标签"""
        return [c["display_name"] for c in result.get("concepts", [])]

    def _extract_citation_trend(self, result: Dict[str, Any]) -> Dict[int, int]:
        """提取最近3年的引用趋势"""
        return {
            item["year"]: item["cited_by_count"]
            for item in result.get("counts_by_year", [])[:3]
        }


if __name__ == "__main__":
    from pprint import pprint

    result = PaperMetaAnalyzer(file_path="examples/Example/pdf/1706.03762v7.pdf")
    pprint(result.metadata)
