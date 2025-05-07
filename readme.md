# Doc_analyze 使用教程

## 1. 创建环境和配置依赖

在终端运行以下命令，创建conda环境并安装依赖：
```
conda env create -f environment.yml
```

创建完成后，可以通过以下命令激活新环境：
```
conda activate Doc_environment
```

如果出现未安装成功的依赖，直接`pip`依次安装缺失的依赖包即可，也可以在`environment.yml`中查看具体的依赖包版本。

## 2. 修改配置文件

打开`config.yaml`进行模型路径的配置，将`predict`,`yolo`,`Structure Processor`下的相关模型路径修改为当前的模型绝对路径。

注意路径是`/`，例如：
```
model_path: D:/Personal_Project/Doc_analyze/yolo_model/best.pt
```

填写`QwenVL`和`OPENAI`的大模型调用API；`OPENAI`采用openai兼容的模型接口。

## 3. 项目使用

将上述文件都配置完成后，运行`main.py`，等待窗口出现后选择上传一个pdf文件，随后会按照流程依次处理。

在`output`文件夹中会生成一个`hashes.json`，会记录已经处理过的pdf文件名称对应的哈希值，如果上传已经处理过的pdf文件，则会显示已被记录不会在被处理，需要在`hashes.json`中删除对应的pdf记录。

`ocr`文件夹用于记录版面分析模型的所有分析结果。

`table_data`文件夹用于记录大模型对表格图片提取的参数信息。

`text_data`文件夹用于记录大模型对文本块提取的参数信息。

`parameter_curve`文件夹用于存放所有被视觉大模型判定为曲线图像的图片。

`parameter.json`会汇总`table_data`和`text_data`中的所有参数数据。

`final_parameter.json`是通过大模型清洗`parameter.json`后的数据。

`final_results`文件夹用于存放曲线提取后的csv文件和png可视化图片。

对于`表格图片提取的参数信息`和`文本块提取的参数信息`的具体结构及其命名规则，可以在`prompt/table_prompt.py`和`prompt/text_prompt.py`中进行修改。

对于`final_parameter.json`的JSON结构和命名规则，可以在`prompt/combine_prompt.py`中修改。