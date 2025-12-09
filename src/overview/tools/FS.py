"""file System for angent."""


def read_file(path: str) -> str | None:
    """Read file with CLI confirmation."""
    confirm = input(f"⚠️  确定要读取文件 '{path}' 吗? (y/n): ")

    # 2. 判断输入，如果不是 y 则终止
    if confirm.lower() not in ("y", ""):
        print("已取消读取操作。")
        return None

    # 3. 执行读取
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data
