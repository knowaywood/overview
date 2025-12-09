from pathlib import Path

from paddleocr import PPStructureV3

_pipeline_instance = None


class Batch:
    def __init__(self, input_dir: str, output_dir: str):

        global _pipeline_instance
        if _pipeline_instance is None:
            _pipeline_instance = PPStructureV3()
            print("模型加载完成")

        self.pdf_to_md(input_dir, output_dir)

    def pdf_to_md(self, input_dir: str, output_dir: str):

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 遍历所有 PDF 文件
        for pdf_file in input_path.glob("*.pdf"):
            print(f"正在处理: {pdf_file.name}")

            # 预测
            output = _pipeline_instance.predict(str(pdf_file))

            # 创建对应的输出文件夹
            doc_output_path = output_path / pdf_file.stem
            doc_output_path.mkdir(parents=True, exist_ok=True)

            markdown_list = []
            markdown_images = []

            # 保存 OCR 结果
            for res in output:
                md_info = res.markdown
                markdown_list.append(md_info)
                markdown_images.append(md_info.get("markdown_images", {}))

            # 拼接 Markdown
            markdown_text = _pipeline_instance.concatenate_markdown_pages(markdown_list)

            # 写入 markdown 文件
            md_file = doc_output_path / f"{pdf_file.stem}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            # 保存所有图片
            for item in markdown_images:
                if item:
                    for img_rel_path, img in item.items():
                        save_path = doc_output_path / img_rel_path
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        img.save(save_path)

            print(f"完成: {pdf_file.name} → {doc_output_path}")

        print("全部转换完成！")


if __name__ == "__main__":
    batch = Batch(
        "/home/aryovel/overview/overview/examples/Example/pdf",
        "/home/aryovel/overview/overview/examples/Example/md",
    )
