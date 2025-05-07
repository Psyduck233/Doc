import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置matplotlib中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei'] #默认字体无法正确渲染某些特殊字符
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def generate_final_results(output_dir, pdf_name):
    """
    生成最终结果的主函数
    :param output_dir: 主输出目录
    :param pdf_name: PDF文件名（不含扩展名）
    """
    yolo_result_dir = Path(output_dir) / "yolo_result"
    final_output_dir = Path(output_dir) / "final_results"
    final_output_dir.mkdir(exist_ok=True)

    print(f"\n开始生成最终结果...")
    print(f"YOLO结果目录: {yolo_result_dir}")
    print(f"最终输出目录: {final_output_dir}")

    processed_folders = 0

    # 遍历所有图表文件夹
    for chart_dir in yolo_result_dir.glob("pdf_page_*_figure_*"):
        if not chart_dir.is_dir():
            continue

        chart_name = chart_dir.name
        print(f"\n处理图表: {chart_name}")

        try:
            # 查找必要的文件
            processed_csv = next(chart_dir.glob("*_processed.csv"), None)
            curve_csv = next(chart_dir.glob("*_curve.csv"), None)

            if not processed_csv or not curve_csv:
                print(f"警告: 缺少必要文件，跳过 {chart_name}")
                continue

            # 解析CSV文件
            processed_data, _ = parse_processed_csv(processed_csv)
            curve_data, _ = parse_curve_csv(curve_csv)

            if not processed_data or not curve_data:
                print(f"警告: 文件解析失败，跳过 {chart_name}")
                continue

            # 标定数据
            calibrated_data = biaoding(processed_data, curve_data)

            # 生成图表和CSV
            huitu(processed_data, calibrated_data, chart_name, str(final_output_dir))

            processed_folders += 1
            print(f"成功生成 {chart_name} 的结果")

        except Exception as e:
            print(f"处理 {chart_name} 时出错: {str(e)}")
            continue

    print(f"\n处理完成！共成功处理 {processed_folders} 个图表")
    print(f"最终结果保存在: {final_output_dir}")

def parse_processed_csv(csv_file_path):
    """
    只负责读取和解析单个 _processed.csv 文件，并返回处理结果。
    返回 (folder_result, error_msg):
        - folder_result: 解析成功时，返回包含提取数据的 list；失败返回 None。
        - error_msg: 出错或特殊提示时，返回字符串；否则返回 None。
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            reader = list(csv.reader(csv_file))  # 转成 list 方便操作

            # 判断行数是否足够
            if len(reader) < 5:
                return None, "does not have enough rows."

            folder_result = []

            # 1) First row: 第二列的第2、3行
            first_row = [reader[1][1], reader[2][1]]

            # 2) Second row: 第2到5行，每行的第8列和第10列
            second_row = []
            for i in range(1, 5):
                if len(reader[i]) >= 10:
                    second_row.extend([reader[i][7], reader[i][9]])

            # 3) Third row: 查找第一列包含 "curve" 的行，并取第二列
            third_row = []
            for row in reader:
                if len(row) >= 2 and "curve" in row[0].lower():
                    third_row.append(row[1])

            # 4) Fourth row: 最后一行第一列内容
            fourth_row = reader[-1][0] if len(reader[-1]) >= 1 else ""

            folder_result.append(first_row)
            folder_result.append(second_row)
            folder_result.append(third_row)
            folder_result.append(fourth_row)

            return folder_result, None

    except Exception as e:
        return None, f"Error processing _processed.csv: {e}"


def parse_curve_csv(csv_file_path):
    """
    只负责读取和解析单个 _curve.csv 文件，返回去掉表头后的所有内容（转换为 Python list）。
    返回 (folder_points, error_msg):
        - folder_points: 若成功解析，则为去掉表头的所有行，每行又是解析后的坐标数据列表；失败则 None。
        - error_msg: 出错或提示时的字符串；否则 None。
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            reader = list(csv.reader(csv_file))

            # 如果只有表头或根本没有数据
            if len(reader) <= 1:
                return None, "has no data rows."

            # 提取表头以外的所有行，并解析每个单元格
            folder_points = []
            for row in reader[1:]:
                parsed_row = []
                for cell in row:
                    try:
                        # 将字符串形式的坐标转换为 Python 列表，比如 "[224, 50]" -> [224, 50]
                        parsed_row.append(eval(cell))
                    except Exception:
                        # 如果某一单元格不是可 eval 的格式，这里可以根据需要处理
                        parsed_row.append(None)
                folder_points.append(parsed_row)

            return folder_points, None

    except Exception as e:
        return None, f"Error processing _curve.csv: {e}"

#########biaoding
def biaoding_normal(processed_data, curve_data):
    """
    Convert pixel coordinates to 2D coordinates for a linear-linear scale.

    Parameters:
        processed_data (list): List containing formatted calibration data.
        curve_data (list): Nested list of pixel coordinates to be converted.

    Returns:
        list of pd.DataFrame: A list of DataFrames, each containing converted 2D coordinates for a curve.
    """
    if len(processed_data) > 1:
        calibration_data = processed_data[1]
    else:
        raise ValueError("processed_data does not have enough rows to extract the second row.")

    # Parse calibration data
    two_d_coords = []
    pixel_coords = []

    for i in range(0, len(calibration_data), 2):
        two_d = float(calibration_data[i])
        pixel = tuple(map(float, calibration_data[i + 1].strip('()').split(',')))
        two_d_coords.append(two_d)
        pixel_coords.append(pixel)

    two_d_coords = np.array(two_d_coords)
    pixel_coords = np.array(pixel_coords)

    # Compute scaling factors and offsets
    scale_x = (two_d_coords[1] - two_d_coords[0]) / (pixel_coords[1, 0] - pixel_coords[0, 0])
    scale_y = (two_d_coords[2] - two_d_coords[3]) / (pixel_coords[2, 1] - pixel_coords[3, 1])

    offset_x = two_d_coords[0] - scale_x * pixel_coords[0, 0]
    offset_y = two_d_coords[2] - scale_y * pixel_coords[2, 1]

    # Convert curve data
    converted_curves = []
    for curve in curve_data:
        curve_2d_coords = [
            (scale_x * x + offset_x, scale_y * y + offset_y) for x, y in curve
        ]
        converted_curves.append(pd.DataFrame(curve_2d_coords, columns=['X_2D', 'Y_2D']))

    print("Biaoding type: normal")
    return converted_curves

def biaoding_ylog(processed_data, curve_data):
    """
    Convert pixel coordinates to 2D coordinates for a linear-logarithmic scale.

    Parameters:
        processed_data (list): List containing formatted calibration data.
        curve_data (list): Nested list of pixel coordinates to be converted.

    Returns:
        list of pd.DataFrame: A list of DataFrames, each containing converted 2D coordinates for a curve.
    """
    if len(processed_data) > 1:
        calibration_data = processed_data[1]
    else:
        raise ValueError("processed_data does not have enough rows to extract the second row.")

    # Parse calibration data
    two_d_coords = []
    pixel_coords = []

    for i in range(0, len(calibration_data), 2):
        two_d = float(calibration_data[i])
        pixel = tuple(map(float, calibration_data[i + 1].strip('()').split(',')))
        two_d_coords.append(two_d)
        pixel_coords.append(pixel)

    two_d_coords = np.array(two_d_coords)
    pixel_coords = np.array(pixel_coords)

    # Compute scaling factors and offsets
    scale_x = (two_d_coords[1] - two_d_coords[0]) / (pixel_coords[1, 0] - pixel_coords[0, 0])
    offset_x = two_d_coords[0] - scale_x * pixel_coords[0, 0]

    log_two_d_y = np.log10(two_d_coords[2:4])
    scale_y = (log_two_d_y[1] - log_two_d_y[0]) / (pixel_coords[3, 1] - pixel_coords[2, 1])
    offset_y = log_two_d_y[0] - scale_y * pixel_coords[2, 1]

    # Convert curve data
    converted_curves = []
    for curve in curve_data:
        curve_2d_coords = [
            (scale_x * x + offset_x, 10 ** (scale_y * y + offset_y)) for x, y in curve
        ]
        converted_curves.append(pd.DataFrame(curve_2d_coords, columns=['X_2D', 'Y_2D']))

    print("Biaoding type: ylog")
    return converted_curves

def biaoding_xlog(processed_data, curve_data):
    """
    Perform calibration to convert pixel coordinates to 2D coordinates.
    The horizontal axis is a logarithmic scale, while the vertical axis is a linear scale.

    Parameters:
        processed_data (list): List containing formatted calibration data.
        curve_data (list): Nested list of pixel coordinates to be converted.

    Returns:
        list of pd.DataFrame: A list of DataFrames, each containing converted 2D coordinates for a curve.
    """
    # Extract the second row from processed_data
    if len(processed_data) > 1:
        calibration_data = processed_data[1]
    else:
        raise ValueError("processed_data does not have enough rows to extract the second row.")

    print("Calibration Data:", calibration_data)

    # Parse calibration data to extract 2D and pixel coordinates
    two_d_coords = []
    pixel_coords = []

    for i in range(0, len(calibration_data), 2):
        try:
            # Parse 2D coordinate (assumed to be a single float for x-log scale)
            two_d = float(calibration_data[i])
            two_d_coords.append(two_d)

            # Parse pixel coordinate (assumed to be a tuple in string format)
            pixel = tuple(map(float, calibration_data[i + 1].strip('()').split(',')))
            pixel_coords.append(pixel)
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid calibration data at index {i}: {e}")

    # Ensure we have enough data for computation
    if len(two_d_coords) < 4 or len(pixel_coords) < 4:
        raise ValueError("Not enough valid calibration data for scaling factor computation.")

    # Convert to numpy arrays
    two_d_coords = np.array(two_d_coords)
    pixel_coords = np.array(pixel_coords)

    # Compute scaling factor for X axis (logarithmic scale)
    log_two_d_x = np.log10(two_d_coords[:2])  # Use logarithmic values of the 2D X coordinates
    scale_x = (log_two_d_x[1] - log_two_d_x[0]) / (pixel_coords[1, 0] - pixel_coords[0, 0])
    offset_x = log_two_d_x[0] - scale_x * pixel_coords[0, 0]

    # Compute scaling factor for Y axis (linear scale)
    scale_y = (two_d_coords[2] - two_d_coords[3]) / (pixel_coords[2, 1] - pixel_coords[3, 1])
    offset_y = two_d_coords[2] - scale_y * pixel_coords[2, 1]

    # Convert each curve's pixel coordinates to 2D coordinates
    converted_curves = []
    for curve in curve_data:
        curve_2d_coords = []
        for pixel_x, pixel_y in curve:
            log_two_d_x = scale_x * pixel_x + offset_x
            two_d_x = 10 ** log_two_d_x  # Convert back from log scale
            two_d_y = scale_y * pixel_y + offset_y
            curve_2d_coords.append((two_d_x, two_d_y))

        # Create a DataFrame for the converted coordinates of this curve
        curve_df = pd.DataFrame(curve_2d_coords, columns=['X_2D', 'Y_2D'])
        converted_curves.append(curve_df)

    print("Biaoding type: xlog")
    return converted_curves

def biaoding_xylog(processed_data, curve_data):
    """
    Convert pixel coordinates to 2D coordinates for a logarithmic-logarithmic scale.

    Parameters:
        processed_data (list): List containing formatted calibration data.
        curve_data (list): Nested list of pixel coordinates to be converted.

    Returns:
        list of pd.DataFrame: A list of DataFrames, each containing converted 2D coordinates for a curve.
    """
    if len(processed_data) > 1:
        calibration_data = processed_data[1]
    else:
        raise ValueError("processed_data does not have enough rows to extract the second row.")

    # Parse calibration data
    two_d_coords = []
    pixel_coords = []

    for i in range(0, len(calibration_data), 2):
        two_d = float(calibration_data[i])
        pixel = tuple(map(float, calibration_data[i + 1].strip('()').split(',')))
        two_d_coords.append(two_d)
        pixel_coords.append(pixel)

    two_d_coords = np.array(two_d_coords)
    pixel_coords = np.array(pixel_coords)

    # Compute scaling factors and offsets
    log_two_d_x = np.log10(two_d_coords[:2])
    scale_x = (log_two_d_x[1] - log_two_d_x[0]) / (pixel_coords[1, 0] - pixel_coords[0, 0])
    offset_x = log_two_d_x[0] - scale_x * pixel_coords[0, 0]

    log_two_d_y = np.log10(two_d_coords[2:4])
    scale_y = (log_two_d_y[1] - log_two_d_y[0]) / (pixel_coords[3, 1] - pixel_coords[2, 1])
    offset_y = log_two_d_y[0] - scale_y * pixel_coords[2, 1]

    # Convert curve data
    converted_curves = []
    for curve in curve_data:
        curve_2d_coords = [
            (10 ** (scale_x * x + offset_x), 10 ** (scale_y * y + offset_y)) for x, y in curve
        ]
        converted_curves.append(pd.DataFrame(curve_2d_coords, columns=['X_2D', 'Y_2D']))

    print("Biaoding type: xylog")
    return converted_curves

def biaoding(processed_data, curve_data):
    """
    该函数的功能是：
    1. 取出 processed_data 的最后一个字符串并打印出来
    2. 根据最后一个字符串判断曲线标定类型，调用相应的标定函数
    3. 打印检测到的标定类型

    :param processed_data: list，包含一些处理后的数据
    :param curve_data: list，原始的曲线数据
    :return: list，标定后的曲线数据
    """
    last_str = processed_data[-1]  # 获取 processed_data 的最后一个元素
    print(f"Last string: {last_str}")  # 打印最后一个字符串

    # 根据最后一个字符串判断曲线标定类型并调用相应的函数
    if "xy_log" in last_str:
        print("Detected type: xy_log")
        curve_data_biaoding = biaoding_xylog(processed_data, curve_data)
    elif "y_log" in last_str:
        print("Detected type: y_log")
        curve_data_biaoding = biaoding_ylog(processed_data, curve_data)
    elif "x_log" in last_str:
        print("Detected type: x_log")
        curve_data_biaoding = biaoding_xlog(processed_data, curve_data)
    else:
        print("Detected type: normal")
        curve_data_biaoding = biaoding_normal(processed_data, curve_data)
    # 打印列表内容
    print("列表内容:")
    print(curve_data_biaoding)

    # 打印元素个数
    print("\n元素个数:")
    print(len(curve_data_biaoding))
    return curve_data_biaoding

def huitu_normal(processed_data, curve_data_new, folder_name, folder_path):
    # 确保文件夹路径存在
    os.makedirs(folder_path, exist_ok=True)

    # 提取x轴和y轴标签
    x_label = processed_data[0][0]
    y_label = processed_data[0][1] if len(processed_data[0]) > 1 else ''

    # 提取曲线标题
    curve_label = processed_data[-2]
    title = f"The extracted curve of {', '.join(curve_label)}"

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    for idx, curve in enumerate(curve_data_new):
        x_coords = curve['X_2D'].tolist()
        y_coords = curve['Y_2D'].tolist()
        plt.plot(x_coords, y_coords, label=f"curve label {idx + 1}")

    # 设置标题和轴标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 保存图表
    png_file_name = f"extract_curve_{folder_name}.png"  # 使用子目录名生成文件名
    png_file_path = os.path.join(folder_path, png_file_name)
    plt.savefig(png_file_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存曲线数据到CSV文件
    csv_file_name = f"extract_curve_{folder_name}.csv"
    csv_file_path = os.path.join(folder_path, csv_file_name)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入标题行
        csv_writer.writerow(["Curve Index", "X_2D", "Y_2D"])
        # 写入每条曲线的数据
        for idx, curve in enumerate(curve_data_new):
            for x, y in zip(curve['X_2D'], curve['Y_2D']):
                csv_writer.writerow([idx + 1, x, y])

    # 打印保存成功消息
    print(f"The plot has been successfully saved at {png_file_path}.")
    print(f"The curve data has been successfully saved at {csv_file_path}.")

def huitu_xlog(processed_data, curve_data_new, folder_name, folder_path):
    # 确保文件夹路径存在
    os.makedirs(folder_path, exist_ok=True)

    # 提取x轴和y轴标签
    x_label = processed_data[0][0]
    y_label = processed_data[0][1] if len(processed_data[0]) > 1 else ''

    # 提取曲线标题
    curve_label = processed_data[-2]
    title = f"The extracted curve of {', '.join(curve_label)}"

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    for idx, curve in enumerate(curve_data_new):
        x_coords = curve['X_2D'].tolist()
        y_coords = curve['Y_2D'].tolist()
        plt.plot(x_coords, y_coords, label=f"curve label {idx + 1}")

    # 设置标题和轴标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xscale('log')

    # 保存图表
    png_file_name = f"extract_curve_{folder_name}.png"  # 使用子目录名生成文件名
    png_file_path = os.path.join(folder_path, png_file_name)
    plt.savefig(png_file_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存曲线数据到CSV文件
    csv_file_name = f"extract_curve_{folder_name}.csv"
    csv_file_path = os.path.join(folder_path, csv_file_name)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入标题行
        csv_writer.writerow(["Curve Index", "X_2D", "Y_2D"])
        for idx, curve in enumerate(curve_data_new):
            for x, y in zip(curve['X_2D'], curve['Y_2D']):
                csv_writer.writerow([idx + 1, x, y])

    print(f"The plot has been successfully saved at {png_file_path}.")
    print(f"The curve data has been successfully saved at {csv_file_path}.")

def huitu_ylog(processed_data, curve_data_new, folder_name, folder_path):
    # 确保文件夹路径存在
    os.makedirs(folder_path, exist_ok=True)

    # 提取x轴和y轴标签
    x_label = processed_data[0][0]
    y_label = processed_data[0][1] if len(processed_data[0]) > 1 else ''

    # 提取曲线标题
    curve_label = processed_data[-2]
    title = f"The extracted curve of {', '.join(curve_label)}"

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    for idx, curve in enumerate(curve_data_new):
        x_coords = curve['X_2D'].tolist()
        y_coords = curve['Y_2D'].tolist()
        plt.plot(x_coords, y_coords, label=f"curve label {idx + 1}")

    # 设置标题和轴标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.yscale('log')

    # 保存图表
    png_file_name = f"extract_curve_{folder_name}.png"  # 使用子目录名生成文件名
    png_file_path = os.path.join(folder_path, png_file_name)
    plt.savefig(png_file_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存曲线数据到CSV文件
    csv_file_name = f"extract_curve_{folder_name}.csv"
    csv_file_path = os.path.join(folder_path, csv_file_name)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入标题行
        csv_writer.writerow(["Curve Index", "X_2D", "Y_2D"])
        for idx, curve in enumerate(curve_data_new):
            for x, y in zip(curve['X_2D'], curve['Y_2D']):
                csv_writer.writerow([idx + 1, x, y])

    print(f"The plot has been successfully saved at {png_file_path}.")
    print(f"The curve data has been successfully saved at {csv_file_path}.")

def huitu_xylog(processed_data, curve_data_new, folder_name, folder_path):
    # 确保文件夹路径存在
    os.makedirs(folder_path, exist_ok=True)

    # 提取x轴和y轴标签
    x_label = processed_data[0][0]
    y_label = processed_data[0][1] if len(processed_data[0]) > 1 else ''

    # 提取曲线标题
    curve_label = processed_data[-2]
    title = f"The extracted curve of {', '.join(curve_label)}"

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    for idx, curve in enumerate(curve_data_new):
        x_coords = curve['X_2D'].tolist()
        y_coords = curve['Y_2D'].tolist()
        plt.plot(x_coords, y_coords, label=f"curve label {idx + 1}")

    # 设置标题和轴标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xscale('log')
    plt.yscale('log')

    # 保存图表
    png_file_name = f"extract_curve_{folder_name}.png"  # 使用子目录名生成文件名
    png_file_path = os.path.join(folder_path, png_file_name)
    plt.savefig(png_file_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存曲线数据到CSV文件
    csv_file_name = f"extract_curve_{folder_name}.csv"
    csv_file_path = os.path.join(folder_path, csv_file_name)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入标题行
        csv_writer.writerow(["Curve Index", "X_2D", "Y_2D"])
        for idx, curve in enumerate(curve_data_new):
            for x, y in zip(curve['X_2D'], curve['Y_2D']):
                csv_writer.writerow([idx + 1, x, y])

    print(f"The plot has been successfully saved at {png_file_path}.")
    print(f"The curve data has been successfully saved at {csv_file_path}.")


def huitu(processed_data, curve_data_new,folder_name,folder_path):
    """
    该函数的功能是：
    1. 取出 processed_data 的最后一个字符串并打印出来
    2. 根据最后一个字符串判断曲线绘图类型，调用相应的绘图函数
    3. 打印检测到的绘图类型

    :param processed_data: list，包含一些处理后的数据
    :param curve_data_new: list，原始的曲线数据
    """
    last_str = processed_data[-1]  # 获取 processed_data 的最后一个元素
    print(f"Last string: {last_str}")  # 打印最后一个字符串

    # 根据最后一个字符串判断曲线标定类型并调用相应的函数
    if "xy_log" in last_str:
        print("Detected type again: xy_log")
        curve_data_biaoding = huitu_xylog(processed_data, curve_data_new,folder_name,folder_path)
    elif "y_log" in last_str:
        print("Detected type again: y_log")
        curve_data_biaoding = huitu_ylog(processed_data, curve_data_new,folder_name,folder_path)
    elif "x_log" in last_str:
        print("Detected type again: x_log")
        curve_data_biaoding = huitu_xlog(processed_data, curve_data_new,folder_name,folder_path)
    else:
        print("Detected type again: normal")
        curve_data_biaoding = huitu_normal(processed_data, curve_data_new,folder_name,folder_path)
