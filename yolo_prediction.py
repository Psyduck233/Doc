import os
from PIL import Image, ImageDraw
import csv
from paddleocr import PaddleOCR
from collections import defaultdict
import pandas as pd
import cv2
import numpy as np
import re
import yaml
from ultralytics import YOLO

# 读取配置文件
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 获取配置文件中的路径
yolo_model_path = config["yolo"]["model_path"]
ocr_det_model = config["yolo"]["ocr_det_model"]
ocr_rec_model = config["yolo"]["ocr_rec_model"]

# 标签对应关系
label_map = {
    0: "x_axis_label",
    1: "y_axis_label",
    2: "x_tick",
    3: "y_tick",
    4: "curve_label",
    5: "x_log_axis",
    6: "y_log_axis",
    7: "xy_log_axis"
}


def run_yolo_prediction(input_dir, output_dir, pdf_name):
    """运行YOLO预测的主函数"""
    # 初始化PaddleOCR
    ocr = PaddleOCR(
        det_model_dir=ocr_det_model,
        rec_model_dir=ocr_rec_model,
        use_angle_cls=False,
        use_space_char=True
    )

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载YOLO模型
    model = YOLO(yolo_model_path)

    # 遍历输入目录中的所有图片
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            print(f"Processing {img_path}")
            results = model.predict(
                source=img_path,
                save=True,
                save_txt=True,
                project=output_dir,
                name=os.path.splitext(img_name)[0],
                conf=0.5
            )

            original_image = Image.open(img_path)
            img_width, img_height = original_image.size

            csv_path = os.path.join(output_dir, os.path.splitext(img_name)[0], f"{os.path.splitext(img_name)[0]}.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            label_save_dir = os.path.join(output_dir, os.path.splitext(img_name)[0], "cropped")
            os.makedirs(label_save_dir, exist_ok=True)

            label_counters = defaultdict(int)
            label_results = defaultdict(list)

            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1 - 10)
                y1 = max(0, y1 - 10)
                x2 = min(img_width, x2 + 10)
                y2 = min(img_height, y2 + 10)

                cropped_image = original_image.crop((x1, y1, x2, y2))
                label_name = label_map.get(int(cls), "unknown")
                label_counters[label_name] += 1
                cropped_image_name = f"{label_name}_{label_counters[label_name]}.png"
                cropped_image_path = os.path.join(label_save_dir, cropped_image_name)
                cropped_image.save(cropped_image_path)

                ocr_result = ocr.ocr(cropped_image_path, cls=False)
                ocr_text = ""
                if ocr_result and isinstance(ocr_result[0], list):
                    ocr_text = " ".join([line[1][0] for line in ocr_result[0]])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_coords = f"({center_x}, {center_y})"

                label_results[label_name].append((cropped_image_name[:-4], ocr_text, center_coords))

            # 创建排序后的结果
            sorted_results = defaultdict(list)
            for label, rows in label_results.items():
                if label in ["x_tick", "y_tick"]:
                    filtered_rows = [row for row in rows if convert_to_number(row[1]) != 0]
                    sorted_rows = sorted(filtered_rows, key=lambda x: convert_to_number(x[1]))
                    sorted_results[label] = sorted_rows
                else:
                    sorted_results[label] = rows

            # 写入CSV文件
            with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    ["Label", "OCR Result", "Center Coordinates", "Label", "OCR Result", "Center Coordinates"])
                for label in label_map.values():
                    if label in label_results:
                        original_rows = label_results[label]
                        sorted_rows = sorted_results[label]
                        for original, sorted_row in zip(original_rows, sorted_rows):
                            if label in ["x_tick", "y_tick"]:
                                sorted_number = convert_to_number(sorted_row[1])
                                csv_writer.writerow(list(original) + [sorted_row[0], sorted_number, sorted_row[2]])
                            else:
                                csv_writer.writerow(list(original) + list(sorted_row))

            # 生成遮罩图像
            mask_labels = {0, 1, 2, 3, 4}
            masked_image = original_image.copy()
            draw = ImageDraw.Draw(masked_image)

            result_folder = os.path.join(output_dir, os.path.splitext(img_name)[0])
            os.makedirs(result_folder, exist_ok=True)

            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                if int(cls) in mask_labels:
                    x1, y1, x2, y2 = map(int, box)
                    draw.rectangle([x1, y1, x2, y2], fill="white")

            mask_image_path = os.path.join(result_folder, f"{os.path.splitext(img_name)[0]}_mask.jpg")
            masked_image.save(mask_image_path)
            print(f"Mask image saved: {mask_image_path}")

    # 处理所有CSV文件
    process_all_csv_in_directory(output_dir)

    # 处理图像获取坐标点
    process_images(input_dir, output_dir)

    print("YOLO prediction and OCR complete! Results saved in:", output_dir)


def convert_to_number(text):
    """OCR转换为数值的函数"""
    multipliers = {'k': 1000, 'm': 1000000, 'g': 1000000000}
    text = text.lower()
    num = ''.join([c if c.isdigit() or c in ['.', '-'] else '' for c in text])
    multiplier = 1
    for key in multipliers:
        if key in text:
            multiplier = multipliers[key]
            break
    try:
        return float(num) * multiplier if num else 0
    except ValueError:
        return 0


def process_all_csv_in_directory(top_directory):
    """处理目录中的所有CSV文件"""
    for root, dirs, files in os.walk(top_directory):
        for file in files:
            if file.endswith(".csv") and not file.endswith("_processed.csv"):
                input_csv_path = os.path.join(root, file)
                df = pd.read_csv(input_csv_path, encoding='utf-8')

                output_csv_path = os.path.join(root, file.replace(".csv", "_processed.csv"))

                value_index = 5
                x_tick_matrices = get_top_bottom_values(df, keyword="x_tick", value_index=value_index)
                y_tick_matrices = get_top_bottom_values(df, keyword="y_tick", value_index=value_index)

                # 确保DataFrame有足够的列
                for i in range(6, 9):
                    if i >= df.shape[1]:
                        df[f'NewCol{i - 5}'] = None

                # 安全赋值，确保长度匹配
                if len(x_tick_matrices) > 0:
                    try:
                        df.iloc[:len(x_tick_matrices), 6:9] = x_tick_matrices[:len(df)]
                    except Exception as e:
                        print(f"处理x_tick_matrices时出错: {e}")
                        continue

                if len(y_tick_matrices) > 0:
                    try:
                        start_idx = len(x_tick_matrices)
                        end_idx = start_idx + len(y_tick_matrices)
                        df.iloc[start_idx:end_idx, 6:9] = y_tick_matrices[:len(df)-start_idx]
                    except Exception as e:
                        print(f"处理y_tick_matrices时出错: {e}")
                        continue

                df.to_csv(output_csv_path, index=False, encoding='utf-8')
                print(f"Processed and saved: {output_csv_path}")
def get_top_bottom_values(df, keyword, value_index):
    """获取顶部和底部值"""
    filtered_df = df[df.iloc[:, 3:6].apply(lambda x: x.astype(str).str.contains(keyword).any(), axis=1)]
    if value_index > df.shape[1]:
        raise IndexError(f"value_index {value_index} 超出了列范围（总列数为 {df.shape[1]}）。")

    column_name = df.columns[value_index - 1]
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    sorted_df = filtered_df.sort_values(by=column_name, ascending=False)

    if len(sorted_df) >= 4:
        selected = sorted_df.iloc[[0, 3]]
    elif len(sorted_df) == 3:
        selected = sorted_df.iloc[[0, -1]]
    elif len(sorted_df) == 2:
        selected = sorted_df.iloc[[0, -1]]
    else:
        selected = sorted_df

    return selected.iloc[:, 3:6].values.tolist()


def get_xy_points(output_path):
    """获取XY坐标点"""
    try:
        mask_images = [f for f in os.listdir(output_path) if f.endswith("_mask.jpg")]
        if not mask_images:
            raise FileNotFoundError(f"在路径 {output_path} 中未找到 '_mask.jpg' 文件")

        image_path = os.path.join(output_path, mask_images[0])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

        horizontal_lines, vertical_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < abs(x2 - x1):
                    horizontal_lines.append((y1 + y2) / 2)
                elif abs(x2 - x1) < abs(y2 - y1):
                    vertical_lines.append((x1 + x2) / 2)

        horizontal_lines = sorted(set(horizontal_lines))
        vertical_lines = sorted(set(vertical_lines))

        if not horizontal_lines or not vertical_lines:
            raise ValueError(f"未检测到足够的直线：{image_path}")

        x_points = [(x, max(horizontal_lines)) for x in vertical_lines]
        y_points = [(min(vertical_lines), y) for y in horizontal_lines]

        return x_points, y_points
    except Exception as e:
        print(f"处理图片时出错: {output_path}, 错误信息: {str(e)}")  # 修改为直接打印错误信息
        return [], []


def process_images(input_dir, output_dir):
    """处理图像"""
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png')):
            print(f"处理图片: {filename}")
            try:
                base_name = os.path.splitext(filename)[0]
                folder_path = os.path.join(output_dir, base_name)

                if not os.path.exists(folder_path):
                    raise FileNotFoundError(f"未找到对应的文件夹：{folder_path}")

                x_points, y_points = get_xy_points(folder_path)
                if not x_points or not y_points:
                    continue

                csv_files = [f for f in os.listdir(folder_path) if f.endswith("_processed.csv")]
                if not csv_files:
                    raise FileNotFoundError(f"未找到 _processed.csv 文件：{folder_path}")

                csv_path = os.path.join(folder_path, csv_files[0])
                df = pd.read_csv(csv_path, header=None, encoding='utf-8')

                def parse_coordinates(value):
                    match = re.match(r"\((\d+),\s*(\d+)\)", str(value))
                    if match:
                        return float(match.group(1)), float(match.group(2))
                    raise ValueError(f"无法解析坐标: {value}")

                try:
                    x01 = parse_coordinates(df.iloc[1, 8])
                    x02 = parse_coordinates(df.iloc[2, 8])
                    y01 = parse_coordinates(df.iloc[3, 8])
                    y02 = parse_coordinates(df.iloc[4, 8])
                except Exception as e:
                    raise ValueError(f"解析 CSV 数据时出错: {csv_path}, 错误信息: {str(e)}")

                def find_closest_point(target, points):
                    return min(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(target)))

                closest_x01 = find_closest_point(x01, x_points)
                closest_x02 = find_closest_point(x02, x_points)
                closest_y01 = find_closest_point(y01, y_points)
                closest_y02 = find_closest_point(y02, y_points)

                new_col_index = 9
                if new_col_index >= df.shape[1]:
                    df = pd.concat([df, pd.DataFrame(columns=range(df.shape[1], new_col_index + 1))], axis=1)

                df.loc[1, new_col_index] = f"({closest_x01[0]}, {closest_x01[1]})"
                df.loc[2, new_col_index] = f"({closest_x02[0]}, {closest_x02[1]})"
                df.loc[3, new_col_index] = f"({closest_y01[0]}, {closest_y01[1]})"
                df.loc[4, new_col_index] = f"({closest_y02[0]}, {closest_y02[1]})"

                df.to_csv(csv_path, index=False, header=False, encoding='utf-8')
                print(f"已更新文件：{csv_path}")
            except Exception as e:
                print(f"处理文件时出错: {filename}, 错误信息: {str(e)}")  # 修改为直接打印错误信息