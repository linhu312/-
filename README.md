train.py是模型训练文件，通过命令python train.py直接运行

jiekou.py是模型的脚本测试文件，通过访问在test文件夹中的图片，模型进行标注之后将生成的的xml文件放在output文件夹中。

OpenVINO 运行时 API .py,顾名思义就是我的best.pt模型转化为onnx格式之后进行模型转化openvino模型优化的代码.

openvinotest.py是我进行模型转化后对模型优化结果的测试推理。

exportmodel.py模型导出文件。

这是我通过yolov10n训练的对于报纸的标注预测的模型的一些训练的代码文件
