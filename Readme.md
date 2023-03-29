## ChatGPT + Stable diffusion
在 [ChuanhuChatGPT](https://github.com/GaiZhenbiao/ChuanhuChatGPT) 旧版的基础上开发，增加了使用ChatGPT与Stable diffusion结合的功能，可以生成角色图像。

## 依赖
- Python 3.9+
- gradio
- openai
- numpy
- tiktoken
- Pillow
- requests

## 使用方法
1. 安装依赖 `pip install -r requirements.txt`
2. 在 Stable diffusion 的启动项中增加 `--api` 
3. 控制台运行 `python main.py` 或者直接打开 `main.bat` 文件(如果你使用了虚拟环境管理python，需要在 `main.bat` 文件中添加虚拟环境的python路径。)
4. （如果需要走代理，可以在 `main.py` 文件中修改 `my_proxy` 变量）
5. 打开127.0.0.1:7988，在API/端口设置中填写你的 OpenAI api Key 和 Stable diffusion 的地址和端口号(一般为127.0.0.1:7860)
6. 打开详细设置-SD相关设置-载入SD模型，然后选择一个模型。
7. 然后就可以点击📷按钮生成图片了。
