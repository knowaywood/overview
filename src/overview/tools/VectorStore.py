import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.tools import Tool
from langchain_text_splitters import MarkdownTextSplitter

load_dotenv()


def get_markdown_files(dir_path: str) -> Set[str]:
    """Retrieve a set of markdown file names from the specified directory.

    Args:
        dir_path (str): The path to the directory to scan.

    Returns:
        Set[str]: A set containing the names of all markdown files found in the directory.

    """
    markdown_files = set()
    try:
        for entry in os.scandir(dir_path):
            if entry.is_file() and entry.name.endswith(".md"):
                markdown_files.add(entry.name)
    except OSError as e:
        print(f"Error: {e}")

    return markdown_files


def add_del_files(data: List[Dict], md_dir: str) -> Tuple[Set[str], Set[str]]:
    """Compare files in the vector store metadata with current markdown files in a directory to determine which files need to be added or deleted from the vector store.

    Args:
        data (List[Dict]): A list of dictionaries, where each dictionary contains metadata
                           about a document, including its 'source' path.
        md_dir (str): The directory containing the current markdown files.

    Returns:
        Tuple[Set[str], Set[str]]: A tuple containing two sets:
                                   - add_files: Files present in md_dir but not in data.
                                   - del_files: Files present in data but not in md_dir (and are markdown files).

    """
    unique_files = set()
    try:
        for item in data:
            source_path = item.get("source")
            if not isinstance(source_path, str):
                print(f"Warning: 'source' is not a string in item {item}. Skipping.")
                continue
            filename = os.path.basename(source_path)
            unique_files.add(filename)
    except Exception as e:
        print(f"Error processing 'data' list: {e}")
    current_md_files = get_markdown_files(md_dir)
    add_files = current_md_files - unique_files
    del_files = {f for f in (unique_files - current_md_files) if f.endswith(".md")}
    return add_files, del_files


class StandVectorStore:
    """A base class for managing a vector store with a DashScope embedding model.It provides functionalities for adding texts and a retriever tool."""

    def __init__(self, name: str = "stand") -> None:
        """Initialize the StandVectorStore with an embedding model and a Chroma vector store.

        Args:
            name (str): The name of the vector store. Defaults to "stand".

        """
        self.embedding_model = DashScopeEmbeddings(model="text-embedding-v3")
        self.vec_store = Chroma(name, self.embedding_model)
        self.retriever = self.vec_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 10}
        )
        self.retriever_tool = Tool(
            name="retriever_tool",
            func=self._retriever_tool_func,
            description="This tool searches and returns the information of math data.",
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Add texts to the vector store without writing to the local file system.

        Args:
            texts (Iterable[str]): The texts to add.
            metadatas (Optional[list[dict]]): Optional metadata for each text.
            ids (Optional[list[str]]): Optional unique IDs for each text.
            **kwargs: Additional keyword arguments.

        """
        self.vec_store.add_texts(texts, metadatas, ids, **kwargs)

    def _retriever_tool_func(self, quary: str) -> str:
        """Internal function for the retriever tool to search and return data information.

        Args:
            quary (str): The query string to search for.

        Returns:
            str: A formatted string of relevant document content or a message if no information is found.

        """
        docs = self.retriever.invoke(quary)
        if not docs:
            return "I found no relevant information in the MathData"
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i + 1} :\n {doc.page_content}")
        return "\n\n".join(results)


class VectorStore(StandVectorStore):
    """Manages a vector store specifically for markdown files, inheriting from StandVectorStore.It provides functionality to synchronize the vector store with a given markdown directory."""

    def __init__(self, md_dir: str, store_dir: str, name: str = "mathdata") -> None:
        """Initialize the VectorStore.

        Args:
            md_dir (str): The directory where markdown files are located.
            store_dir (str): The directory where the vector store data will be persisted.
            name (str): The name of the vector store. Defaults to "mathdata".

        """
        super().__init__()
        self.md_dir = md_dir
        self.store_dir = store_dir
        self.vec_store = Chroma(name, self.embedding_model, persist_directory=store_dir)
        self.sync_store()

    def sync_store(self) -> None:
        """Synchronize the vector store with the markdown files in the specified directory. It identifies new files to add and existing files to delete from the store."""
        try:
            all_metadata = self.vec_store.get(include=["metadatas", "ids"])
        except Exception:
            all_metadata = {"metadatas": [], "ids": []}
        metadata_docs = all_metadata["metadatas"]
        adds, dels = add_del_files(metadata_docs, self.md_dir)
        if adds:
            loader = DirectoryLoader(
                self.md_dir,
                glob=list(adds),  # Ensure glob patterns are full paths
                loader_cls=UnstructuredMarkdownLoader,
                use_multithreading=True,
            )
            doc_pages = loader.load()
            text_spliter = MarkdownTextSplitter(chunk_size=150, chunk_overlap=20)
            doc_chunk = text_spliter.split_documents(doc_pages)
            print(len(doc_chunk), doc_chunk[0] if doc_chunk else None)
            self.vec_store.add_documents(doc_chunk)
            print(f"vec_store added: {adds}\n")
        if dels:
            matching_ids = [
                current_id
                for current_id, current_meta in zip(
                    all_metadata["ids"], all_metadata["metadatas"], strict=False
                )
                if os.path.basename(current_meta["source"]) in dels
            ]
            self.vec_store.delete(ids=matching_ids)
            print(f"vec_store deleted: {dels}\n")
        print("sync_store completed")
