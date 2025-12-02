"""using the packages."""

from dotenv import load_dotenv

from overview.tools.VectorStore import VectorStore

load_dotenv()
vec_store = VectorStore(
    md_dir="./examples/Example/md",
    store_dir="./examples/Example/vectStore",
)
