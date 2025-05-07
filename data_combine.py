import os
import json


def combine_json_data(pdf_name: str):
    table_data_dir = os.path.join("output", pdf_name, "table_data")
    text_data_dir = os.path.join("output", pdf_name, "text_data")
    output_file = os.path.join("output", pdf_name, "parameter.json")

    # 汇总所有JSON数据
    device_names = []  # 用于存储所有器件名称
    parameters = {}

    # 读取 table_data 中的 JSON 文件
    for fname in os.listdir(table_data_dir):
        if fname.lower().endswith(".json"):
            table_json_path = os.path.join(table_data_dir, fname)
            with open(table_json_path, "r", encoding="utf-8") as f:
                table_data = json.load(f)
                if table_data:
                    # 汇总器件名称
                    device_name = table_data.get("Device name", {}).get("name", "")
                    if device_name and device_name not in device_names:
                        device_names.append(device_name)

                    # 汇总参数
                    table_parameters = table_data.get("Parameters", {})
                    for param, value in table_parameters.items():
                        if param not in parameters:
                            parameters[param] = value
                        else:
                            # 合并参数的多个值
                            if isinstance(parameters[param], list):
                                if value not in parameters[param]:
                                    parameters[param].append(value)
                            else:
                                parameters[param] = [parameters[param], value]

    # 读取 text_data 中的 JSON 文件
    for fname in os.listdir(text_data_dir):
        if fname.lower().endswith(".json"):
            text_json_path = os.path.join(text_data_dir, fname)
            with open(text_json_path, "r", encoding="utf-8") as f:
                text_data = json.load(f)
                if text_data:
                    # 汇总器件名称
                    device_name = text_data.get("Device name", {}).get("name", "")
                    if device_name and device_name not in device_names:
                        device_names.append(device_name)

                    # 汇总参数
                    text_parameters = text_data.get("Parameters", {})
                    for param, value in text_parameters.items():
                        if param not in parameters:
                            parameters[param] = value
                        else:
                            # 合并参数的多个值
                            if isinstance(parameters[param], list):
                                if value not in parameters[param]:
                                    parameters[param].append(value)
                            else:
                                parameters[param] = [parameters[param], value]

    # 合并最终的 JSON 数据
    final_output = {
        "Device name": {
            "name": " ".join(device_names)
        },
        "Parameters": parameters
    }

    # 输出最终合并的 JSON 数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(final_output, f_out, ensure_ascii=False, indent=2)

    print(f"✅ 汇总数据已保存至: {output_file}")
