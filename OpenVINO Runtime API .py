import cv2
import numpy as np
from openvino.inference_engine import IECore

# 加载模型
def load_model(xml_path, bin_path):
    ie = IECore()
    # 读取网络结构和权重
    net = ie.read_network(model=xml_path, weights=bin_path)
    # 加载网络到设备（例如 CPU）
    exec_net = ie.load_network(network=net, device_name="CPU")
    return exec_net, net

# 预处理图片
def preprocess_image(image_path, input_shape):
    # 读取图像并调整大小
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_shape[3], input_shape[2]))  # (H, W)
    
    # 转换为 RGB 和归一化
    image = image[..., ::-1]  # BGR to RGB
    image = image.astype(np.float32)
    image -= 127.5
    image /= 127.5
    
    # 添加批次维度
    image = np.expand_dims(image, axis=0)
    return image

# 执行推理
def inference(exec_net, input_blob, image):
    # 执行推理
    res = exec_net.infer(inputs={input_blob: image})
    return res

# 解析结果
def postprocess_result(result, classes):
    # 假设你是做分类任务的，取预测概率最大的类别
    predictions = result[next(iter(result))]  # 获取模型输出
    class_id = np.argmax(predictions)
    return classes[class_id]

def main():
    # 模型路径
    xml_path = "/root/yolov10-main/out/best.xml"
    bin_path = "/root/yolov10-main/out/best.bin"
    
    # 加载模型
    exec_net, net = load_model(xml_path, bin_path)
    
    # 输入层名称和形状
    input_blob = next(iter(net.input_info))  # 获取输入层名称
    input_shape = net.input_info[input_blob].tensor_desc.dims  # 获取输入层形状
    
    # 类别列表（根据你的数据集更新）
    classes = ["Header", "Title", "Text", "Figure", "Foot"]
    
    # 图像路径
    image_path = "/root/yolov10-main/test/001.jpg"
    
    # 预处理图像
    image = preprocess_image(image_path, input_shape)
    
    # 执行推理
    result = inference(exec_net, input_blob, image)
    
    # 处理结果
    predicted_class = postprocess_result(result, classes)
    
    # 输出预测结果
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
