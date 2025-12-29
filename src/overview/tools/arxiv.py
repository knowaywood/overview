""" "download paper from arxiv."""

import os
from typing import Literal, TypedDict

import feedparser
import requests
from feedparser.util import FeedParserDict

from overview import config as cfg


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
    """Download arxiv paper to local hard drive based on paper url.

    Args:
        pdf_url (str): papar url
        save_dir (str, optional): the path to save. Defaults to "./download".
        filename (str | None, optional): file name . Defaults to None.

    Raises:
        RuntimeError: _description_

    Returns:
        str: the path of saved papar

    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not filename:
        name = pdf_url.split("/")[-1]
        if not name.endswith(".pdf"):
            name += ".pdf"
        filename = name

    save_path = os.path.join(save_dir, filename)
    abs_save_path = os.path.abspath(save_path)
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
        cfg.paper_dowload.append(abs_save_path)
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


class DDGRes(TypedDict):
    """the return type of DDGSearcher.search."""

    snippet: str
    title: str
    link: str


class DDGSearcher:
    def __init__(self, query: str, max_results: int = 5) -> None:
        self.query = query
        self.max_results = max_results
        self.answer = self.search(query, max_results)

    @classmethod
    def search(cls, query: str, max_results: int = 5) -> list[DDGRes]:
        from langchain_community.tools import DuckDuckGoSearchResults

        search = DuckDuckGoSearchResults(output_format="list", num_results=max_results)
        query = query + " site:arxiv.org filetype:pdf"
        return search.invoke(query)

    def _get_arxiv_id(self, DDGanswer: list[DDGRes]) -> tuple[list[str], BasePaperInfo]:
        import re
        try:
            if not DDGanswer:
                raise ValueError("No result found.")
            arxiv_url_ls = [i["link"] for i in DDGanswer if "arxiv.org" in i["link"] and ".pdf" in i["link"]]
            print(arxiv_url_ls)
            if not arxiv_url_ls:
                raise ValueError("No arXiv URL found.")
            url = arxiv_url_ls[0]
            arxiv_id_list = []
            for url in arxiv_url_ls:
                match = re.search(r'/pdf/([\d.]+)(?:\.pdf)?$', url)
                if match:
                    arxiv_id_list.append(match.group(1))

            if not arxiv_id_list:
                raise ValueError("No ArXiv ID in the links.")
            
            first_id = arxiv_id_list[0]
            paper_info = self._search_by_id(first_id)
            
            return arxiv_id_list, paper_info


        except ValueError:
            return [], BasePaperInfo()
        

    @classmethod
    def _search_by_id(cls,arxiv_id:str)->BasePaperInfo:

        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        try:
            response = requests.get(api_url, timeout=20)
            response.raise_for_status()

            feed = feedparser.parse(response.text)
            if not feed.entries:
                raise ValueError(f"无法找到ArXiv ID对应的论文: {arxiv_id}")

            entry = feed.entries[0]


            authors = [author.name for author in entry.authors] if hasattr(entry, "authors") else []

            pdf_url = ""
            for link in entry.links:
                if link.type == "application/pdf":
                    pdf_url = link.href
                    break

            paper_info = BasePaperInfo(
                title=entry.title,
                authors=authors,
                summary=entry.summary,
                pdf_url=pdf_url,
                arxiv_id=arxiv_id,
                publish_time=entry.published if hasattr(entry, "published") else "Unknown"
            )

            return paper_info

        except Exception as e:
            raise ValueError(f"搜索过程中出现错误: {e}")

if __name__ == "__main__":
    search = DDGSearcher("machine learning")
    print(search.answer)
    print(search._get_arxiv_id(search.answer))
