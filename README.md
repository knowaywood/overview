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
