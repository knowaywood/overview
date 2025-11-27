

import requests
import feedparser
import os
import argparse
import time


class ArxivSearcher:
    def __init__(self, save_dir="D:\\PyCharm Community Edition 2024.2.1\\pythonProject\\downloads"):
        self.base_url = "http://export.arxiv.org/api/query"
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"[+] 创建下载目录: {save_dir}")

    def search(self, query, max_results=5):
        url = (
            f"{self.base_url}?search_query={query}"
            f"&start=0&max_results={max_results}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )

        print(f"[+] 搜索查询: {query}")
        print(f"[+] 请求URL: {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()
            print("[√] API请求成功")

            feed = feedparser.parse(response.text)
            print(f"[+] 找到 {len(feed.entries)} 篇论文")

            results = []

            for entry in feed.entries:
                pdf_link = ""
                for link in entry.links:
                    if link.type == "application/pdf":
                        pdf_link = link.href
                        break

                arxiv_id = entry.id.split('/')[-1]

                results.append({
                    "title": entry.title,
                    "authors": [author.name for author in entry.authors],
                    "summary": entry.summary,
                    "pdf_url": pdf_link,
                    "arxiv_id": arxiv_id,
                    "published": entry.published if hasattr(entry, 'published') else "Unknown"
                })

            return results

        except Exception as e:
            print(f"[!] 搜索过程中出现错误: {e}")
            return []

    def download_pdf(self, pdf_url, filename=None):

        if not filename:
            name = pdf_url.split("/")[-1]
            if not name.endswith(".pdf"):
                name += ".pdf"
            filename = name

        save_path = os.path.join(self.save_dir, filename)

        print(f"[+] 开始下载: {pdf_url}")
        print(f"[+] 保存路径: {save_path}")

        try:
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

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
            return None


def download_deep_learning_paper():

    download_dir = r"D:\PyCharm Community Edition 2024.2.1\pythonProject\deep_learning_papers"

    searcher = ArxivSearcher(save_dir=download_dir)

    print("=" * 60)
    print("ArXiv 深度学习论文搜索与下载工具")
    print("=" * 60)


    search_query = "deep learning"
    max_results = 3


    results = searcher.search(search_query, max_results=max_results)

    if not results:
        return

    # 显示搜索结果
    print(f"\n[+] 找到 {len(results)} 篇相关论文:")
    print("-" * 60)

    for i, paper in enumerate(results, 1):
        print(f"\n📄 论文 #{i}:")
        print(f"   标题: {paper['title']}")
        print(f"   作者: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
        print(f"   发布时间: {paper['published']}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        print(f"   摘要预览: {paper['summary'][:200]}...")
        print("-" * 60)

    # 下载第一篇论文
    if results:
        first_paper = results[0]
        print(f"\n[+] 正在下载第一篇论文: {first_paper['title']}")

        # 使用arXiv ID作为文件名
        filename = f"{first_paper['arxiv_id']}.pdf"
        downloaded_file = searcher.download_pdf(first_paper['pdf_url'], filename=filename)

        if downloaded_file:
            print(f" 下载成功!")

        else:
            print("[!] 下载失败")





if __name__ == "__main__":
    # 直接运行脚本时，默认下载深度学习论文
    download_deep_learning_paper()