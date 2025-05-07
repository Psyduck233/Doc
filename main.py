import os
import hashlib
import json
from tkinter import filedialog, Tk
from pdf_to_images import pdf_to_images
from structure_processor import process_structure_batch
from table_llm import run_table_extraction
from figure_llm import run_figure_analysis
from text_llm import run_text_extraction
from data_combine import combine_json_data
from combine_llm import run_combine_llm
from predict import run_predict
from yolo_prediction import run_yolo_prediction
from merge import merge_results
from generation import generate_final_results  # 新增导入

# 获取哈希映射的存储路径
HASH_MAPPING_FILE = "output/hashes.json"


def generate_pdf_hash(pdf_path):
    """根据PDF文件路径生成哈希值"""
    return hashlib.md5(pdf_path.encode('utf-8')).hexdigest()


def load_hash_mapping():
    """加载哈希映射文件，如果文件不存在则返回空字典"""
    if os.path.exists(HASH_MAPPING_FILE):
        with open(HASH_MAPPING_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_hash_mapping(mapping):
    """保存哈希映射到文件"""
    with open(HASH_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 选择 PDF 文件
    root = Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(title="选择 PDF 文件", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        print("未选择文件，程序终止。")
        exit()

    # 生成哈希值
    pdf_hash = generate_pdf_hash(pdf_path)

    # 加载现有的哈希映射
    hash_mapping = load_hash_mapping()

    # 检查是否已处理该 PDF，如果处理过则跳过
    if pdf_hash in hash_mapping:
        print("该 PDF 文件已经处理过，跳过。")
        exit()

    # 添加新的哈希映射
    hash_mapping[pdf_hash] = os.path.basename(pdf_path)

    # 保存哈希映射
    save_hash_mapping(hash_mapping)

    # 使用哈希值创建文件路径
    output_dir = os.path.join("output", pdf_hash)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 执行 PDF 转换为图片
    pdf_name, image_paths = pdf_to_images(pdf_path)
    pages_dir = os.path.join(output_dir, "pages")
    ocr_output_dir = os.path.join(output_dir, "ocr")

    # 执行结构化分析
    process_structure_batch(pages_dir, ocr_output_dir)

    # 表格图像分析
    run_table_extraction(pdf_name)

    # 文本数据抽取
    run_text_extraction(pdf_name)

    # 合并table_data和text_data中的json文件
    combine_json_data(pdf_name)

    # 运行combine_llm进行参数清洗并生成最终的parameter.json
    combine_json_file_path = os.path.join(output_dir, "parameter.json")
    run_combine_llm(combine_json_file_path)

    # 图像曲线判断分析
    run_figure_analysis(pdf_name)

    # 运行 LineFormer 模型进行图像预测
    predict_input_dir = os.path.join(output_dir, "parameter_curve")  # 从 figure_llm 输出的路径
    run_predict(predict_input_dir, pdf_name)

    # 运行 YOLO 预测
    yolo_input_dir = os.path.join(output_dir, "parameter_curve")  # 与 LineFormer 相同的输入目录
    yolo_output_dir = os.path.join(output_dir, "yolo_result")
    run_yolo_prediction(yolo_input_dir, yolo_output_dir, pdf_name)

    # 合并 LineFormer 和 YOLO 的结果
    merge_results(output_dir, pdf_name)

    # 生成最终结果（图表和CSV）
    generate_final_results(output_dir, pdf_name)