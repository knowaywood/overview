"""file System for angent."""


def read_from_local(path: str) -> str | None:
    """Read file from local hard drive with CLI confirmation."""
    confirm = input(f"⚠️  确定要读取文件 '{path}' 吗? (y/n): ")

    # 2. 判断输入，如果不是 y 则终止
    if confirm.lower() not in ("y", ""):
        print("已取消读取操作。")
        return None

    # 3. 执行读取
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def save2local(path: str, context: str) -> None:
    """Save file to local hard drive with CLI confirmation.

    Args:
        path (str): file path
        context (str): context to save

    Returns:
        None: _description_

    """
    confirm = input(f"⚠️  确定要写入文件 '{path}' 吗? (y/n): ")

    # 2. 判断输入，如果不是 y 则终止
    if confirm.lower() not in ("y", ""):
        print("已取消写入操作。")
        return None
    # 写入文件
    with open(path, "w", encoding="utf-8") as f:
        f.write(context)
        print(f"✅ 文件 '{path}' 写入成功。")


def save2local_no_confirm(path: str, context: str) -> None:
    """Save file to local hard drive with CLI confirmation.

    Args:
        path (str): file path
        context (str): context to save

    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(context)
        print(f"✅ 文件 '{path}' 写入成功。")


if __name__ == "__main__":
    save_file("test.txt", "hello world")
