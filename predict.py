import os
import sys
import cv2
import csv
import yaml
from pathlib import Path

# 读取配置文件
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 获取配置路径
lineformer_dir = config["predict"]["lineformer_dir"]
ckpt = config["predict"]["predict_ckpt"]
config_file = config["predict"]["predict_config"]

# 添加路径到系统路径
sys.path.insert(0, lineformer_dir)

# 导入LineFormer模块
try:
    import infer
    import line_utils
except ImportError as e:
    raise ImportError(f"无法导入LineFormer模块: {e}")

# 加载模型
infer.load_model(config_file, ckpt, "cpu")


def run_predict(predict_input_dir, pdf_name):
    """主预测函数"""
    # 准备输出目录
    output_base = Path("output") / pdf_name
    csv_dir = output_base / "predict_result" / "csv"
    img_dir = output_base / "predict_result" / "images"

    # 创建输出目录
    csv_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    processed_files = []

    print(f"\n开始处理图表图像...")
    print(f"输入目录: {predict_input_dir}")
    print(f"CSV输出目录: {csv_dir}")
    print(f"图像输出目录: {img_dir}")

    # 处理每个图像文件
    for img_file in Path(predict_input_dir).glob("*.*"):
        if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue

        # 验证文件名格式
        if not ("pdf_page_" in img_file.stem and "_figure_" in img_file.stem):
            print(f"跳过非标准文件名: {img_file.name}")
            continue

        try:
            # 读取图像
            img = cv2.imread(str(img_file))
            if img is None:
                raise ValueError("无法读取图像文件")

            # 运行预测
            line_dataseries = infer.get_dataseries(img, to_clean=False)
            points_array = line_utils.points_to_array(line_dataseries)

            # 保存CSV (使用与输入文件相同的基础名)
            csv_path = csv_dir / f"{img_file.stem}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                writer.writerows(points_array)

            # 保存预测图像 (确保使用_predicted后缀)
            predicted_img = line_utils.draw_lines(img, points_array)
            img_path = img_dir / f"{img_file.stem}_predicted.png"
            cv2.imwrite(str(img_path), predicted_img)

            processed_files.append(img_file.name)
            print(f"已处理: {img_file.name} -> {img_path.name}")

        except Exception as e:
            print(f"处理失败 {img_file.name}: {str(e)}")
            continue

    # 输出摘要
    print(f"\n处理完成！共处理 {len(processed_files)} 个图表")
    print(f"CSV文件保存在: {csv_dir}")
    print(f"预测图像保存在: {img_dir}")