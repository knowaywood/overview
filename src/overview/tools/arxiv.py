""" "download paper from arxiv."""

import os
from typing import Literal, TypedDict

import feedparser
import requests
from feedparser.util import FeedParserDict


class BasePaperInfo(TypedDict):
    """basic infomation of paper."""

    title: list[FeedParserDict]
    authors: list[list[FeedParserDict]]
    summary: list[FeedParserDict]
    pdf_url: list[FeedParserDict] | Literal[""]
    arxiv_id: str
    publish_time: list[FeedParserDict] | Literal["Unknown"]


def download_url(
    pdf_url: str, save_dir: str = "./download", filename: str | None = None
) -> str:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not filename:
        name = pdf_url.split("/")[-1]
        if not name.endswith(".pdf"):
            name += ".pdf"
        filename = name

    save_path = os.path.join(save_dir, filename)

    print(f"[+] 开始下载: {pdf_url}")
    print(f"[+] 保存路径: {save_path}")

    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(save_path, "wb") as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for data in response.iter_content(chunk_size=8192):
                    downloaded += len(data)
                    f.write(data)
        return save_path

    except Exception as e:
        raise RuntimeError(f"下载失败（URL: {pdf_url}, 文件名: {filename}: {e}")


def download_info(
    pdf_info: BasePaperInfo, save_dir: str = "download", filename: str | None = None
):
    return download_url(pdf_info["pdf_url"], save_dir, filename)


class ArxivSearcher:
    """Search and download papers from arXiv."""

    def __init__(self, query: str, max_results: int = 5) -> None:
        """Initialize the ArxivSearcher."""
        self.base_url = "http://export.arxiv.org/api/query"
        self.query = query
        self.max_results = max_results
        self.answer = self.search(query, max_results)

    @staticmethod
    def search(query: str, max_results: int = 5) -> list[BasePaperInfo]:
        """Search papers from arXiv.

        Args:
            query: Search query for papers
            max_results: Maximum number of results to return

        Returns:
            List of paper information dictionaries

        """
        url = (
            f"http://export.arxiv.org/api/query?search_query={query}"
            f"&start=0&max_results={max_results}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )

        print(f"[+] 搜索查询: {query}")
        print(f"[+] 请求URL: {url}")

        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            print("[√] API请求成功")

            feed = feedparser.parse(response.text)

            results = []

            for entry in feed.entries:
                pdf_link = ""
                for link in entry.links:
                    if link.type == "application/pdf":
                        pdf_link = link.href
                        break

                arxiv_id = entry.id.split("/")[-1]
                paperInfo = BasePaperInfo(
                    title=entry.title,
                    authors=[author.name for author in entry.authors],
                    summary=entry.summary,
                    pdf_url=pdf_link,
                    arxiv_id=arxiv_id,
                    publish_time=entry.published
                    if hasattr(entry, "published")
                    else "Unknown",
                )
                results.append(paperInfo)

            return results

        except Exception as e:
            print(f"[!] 搜索过程中出现错误: {e}")
            return []

    def __getitem__(self, index: int) -> BasePaperInfo:
        """Get paper info by index."""
        return self.answer[index]

    def __str__(self) -> str:
        """Return a string representation of the search results."""
        if not self.answer:
            return "No results found"

        res = f"\n[+] Found {len(self.answer)} relevant papers:\n" + "-" * 60

        for i, paper in enumerate(self.answer, 1):
            authors_str = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors_str += "..."

            res += (
                f"\n📄 Paper #{i}:\n"
                f"   Title: {paper['title']}\n"
                f"   Authors: {authors_str}\n"
                f"   Published: {paper['publish_time']}\n"
                f"   arXiv ID: {paper['arxiv_id']}\n"
                f"   Summary: {paper['summary']}\n" + "-" * 60
            )
        return res


if __name__ == "__main__":
    search = ArxivSearcher("deep learning")
    download_info(search[0])
