"""using the packages."""

from overview.tools.VectorStore import VectorStore

vec_store = VectorStore(
    md_dir="./examples/Example/md",
    store_dir="./examples/Example/vectStore",
)
