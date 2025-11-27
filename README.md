首先使用命令（在windows）

```bash
git clone < this repo >

cd langgraph_lr

move .env_copy .env
```

然后在.env中配置你的环境变量，你可以在文件中的url获取你的api key

下载依赖：

```bash
uv sync
```

注意：
- 使用下列代码生成graph png

```python
from PIL import Image
import io

mermaid_png = app.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(mermaid_png))
image.save("mermaid_graph.png")
```
其中app为compile后的graph

## TODO：

- [ ] OCR:  pdf => .md  包括如何部署模型以及遇到的问题。
- [ ] prompt: 研究如何写prompt （比较简单），
- [ ] search tool: 使用arXiv api获取到pdf或其他格式
- [ ] paper模块分析: 分析论文的模块、格式，以及如何分配模块的weight（需要分享）

## PP-structureV3 本地部署
使用wsl:Ubuntu作为运行环境,或参考[链接](https://www.paddleocr.ai/v3.0.0/version2.x/ppocr/environment.html)配置环境

安装NVIDIA驱动与CUDA

创建虚拟环境
```bash

conda create --name paddle_env python=3.9 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda activate paddle_env
```

安装paddle (以CUDA11.8为例,其他版本参考[链接](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html))
```bash
 python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
 pip install "paddlex[ocr]==3.3.9"
 ```
创建运行脚本
```python
# run.py
from pathlib import Path
from paddleocr import PPStructureV3

input_file = "your/input/directory"
output_path = Path("./output")

pipeline = PPStructureV3()
output = pipeline.predict("your/input/directory")

markdown_list = []
markdown_images = []

for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)
```
运行
```bash
python3 ./run.py
```