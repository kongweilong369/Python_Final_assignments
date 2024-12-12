from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import json
import urllib.request

# 创建 Flask 应用
app = Flask(__name__)

# 下载 ImageNet 类别标签
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urllib.request.urlopen(url) as response:
    class_idx = json.load(response)

# 提取类别名称
classes = [class_idx[str(k)][1] for k in range(1000)]

# 加载 ViT 模型
model = timm.create_model('vit_base_patch16_224', pretrained=True)  # 使用预训练的 ViT 模型
model.eval()  # 设置为评估模式

# 渲染前端 HTML 页面
@app.route('/')
def index():
    return render_template('index.html')  # 渲染模板文件

# 处理图片并返回预测结果
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # 对图片进行预处理，并添加批次维度
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(image)  # 将图像输入模型
        print(outputs.shape)  # 打印输出形状（用于调试）
        _, predicted_class = torch.max(outputs, 1)  # 获取最大概率的类别索引

    # 获取类别名称
    predicted_class_name = classes[predicted_class.item()]

    return jsonify({'prediction': predicted_class_name})

# 启动 Flask 应用
if __name__ == '__main__':
    app.run(debug=True)
