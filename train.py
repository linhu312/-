# coding:utf-8
from ultralytics import YOLOv10
# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/v10/yolov10n.yaml"
# 数据集配置文件
data_yaml_path = '/root/yolov10-main/dataset/data.yaml'
# 预训练模型
pre_model_name = 'yolov10n.pt'

if __name__ == '__main__' :
#加载预训练模型

    model = YOLOv10("ultralytics/cfg/models/v10/yolov10n.yaml").load('yolov10n.pt')
# 训练模型
    results = model.train(data=data_yaml_path,epochs=200,batch=16,name='train_v10',device=0)