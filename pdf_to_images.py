import os
from pdf2image import convert_from_path
import hashlib
import yaml

def load_config(config_file="config.yaml"):
    """加载 YAML 配置文件"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件 {config_file} 不存在，请检查路径是否正确。")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_pdf_hash(pdf_path):
    """根据PDF文件路径生成哈希值"""
    return hashlib.md5(pdf_path.encode('utf-8')).hexdigest()


def pdf_to_images(pdf_path, config_file="config.yaml"):
    # 加载配置文件
    config = load_config(config_file)
    poppler_path = config.get("pdf2image", {}).get("poppler_path")

    if not poppler_path or not os.path.isdir(poppler_path):
        raise ValueError(f"无效的 Poppler 路径：{poppler_path}，请检查配置文件。")

    # 使用哈希值作为文件夹名称
    pdf_hash = generate_pdf_hash(pdf_path)
    output_folder = os.path.join("output", pdf_hash, "pages")
    os.makedirs(output_folder, exist_ok=True)

    print(f"📄 转换中... PDF路径：{pdf_path}")
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

    image_paths = []
    for i, img in enumerate(images):
        img_filename = f"pdf_page_{i + 1}.png"
        full_path = os.path.join(output_folder, img_filename)
        img.save(full_path, "PNG")
        image_paths.append(full_path)
        print(f"✅ 已保存：{img_filename}")

    print(f"📂 所有页面图像保存在：{output_folder}")
    return pdf_hash, image_paths