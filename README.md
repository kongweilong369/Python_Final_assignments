项目文件结构
my_project/

│

├── flask_server.py  # Flask 后端代码

├── templates/

│   └── index.html  # 前端 HTML 页面

└── static/          # 静态文件（如 CSS, JS, 图片）

    └── style.css    # 可选的样式文件
    
采用的模型为VIT_Base，数据集为image-net，分类类别有1000种。
这段代码实现了一个简单的 Flask Web 应用，用户可以通过浏览器上传图像，服务器使用预训练的 Vision Transformer (ViT) 模型对图像进行分类，并返回预测的类别名称。
