import os
import shutil
from pathlib import Path


def merge_results(output_dir, pdf_name):
    """合并LineFormer和YOLO的结果"""
    # 准备路径
    base_dir = Path(output_dir)
    yolo_dir = base_dir / "yolo_result"
    predict_csv_dir = base_dir / "predict_result" / "csv"
    predict_img_dir = base_dir / "predict_result" / "images"

    print(f"\n开始合并结果...")
    print(f"YOLO目录: {yolo_dir}")
    print(f"LineFormer CSV目录: {predict_csv_dir}")
    print(f"LineFormer图片目录: {predict_img_dir}")

    success_count = 0

    # 遍历YOLO结果目录
    for yolo_folder in yolo_dir.glob("pdf_page_*_figure_*"):
        if not yolo_folder.is_dir():
            continue

        base_name = yolo_folder.name
        print(f"\n处理图表: {base_name}")

        # 复制CSV文件
        csv_src = predict_csv_dir / f"{base_name}.csv"
        csv_dst = yolo_folder / f"{base_name}_curve.csv"

        if csv_src.exists():
            shutil.copy(csv_src, csv_dst)
            print(f"已复制CSV: {csv_dst}")
            success_count += 1
        else:
            print(f"警告: 未找到CSV文件: {csv_src}")

        # 复制图片文件
        img_src = predict_img_dir / f"{base_name}_predicted.png"
        img_dst = yolo_folder / f"{base_name}_LineFormer.png"

        if img_src.exists():
            shutil.copy(img_src, img_dst)
            print(f"已复制图片: {img_dst}")
            success_count += 1
        else:
            print(f"警告: 未找到图片文件: {img_src}")

    # 输出摘要
    print(f"\n合并完成！成功合并 {success_count // 2} 个图表的结果")