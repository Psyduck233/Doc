import os
import yaml
import numpy as np
from PIL import Image
from paddleocr import PPStructure, draw_structure_result
import json
import codecs

# 载入配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)['Structure Processor']

# OCR 模型配置
ocr_config = {
    'layout_model_dir': config['layout_model_dir'],
    'layout_dict_path': config['layout_dict_path'],
    'table_model_dir': config['table_model_dir'],
    'formula_model_dir': config['formula_model_dir'],
    'det_model_dir': config['det_model_dir'],
    'rec_model_dir': config['rec_model_dir'],
    'show_log': True
}
font_path = config['font_path']
table_engine = PPStructure(**ocr_config)

def process_structure_batch(pages_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化集中输出目录
    figure_output_dir = os.path.join(output_dir, "figure")
    vis_output_dir = os.path.join(output_dir, "visualization")
    table_output_dir = os.path.join(output_dir, "table")
    text_output_dir = os.path.join(output_dir, "text")  # ✅ 新增：文本块汇总输出
    os.makedirs(figure_output_dir, exist_ok=True)
    os.makedirs(vis_output_dir, exist_ok=True)
    os.makedirs(table_output_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)  # ✅ 创建 text 文件夹

    image_files = [f for f in os.listdir(pages_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_name in sorted(image_files):
        image_path = os.path.join(pages_dir, image_name)
        print(f"\n📄 正在处理: {image_path}")

        try:
            pil_image = Image.open(image_path).convert('RGB')
            img = np.array(pil_image)

            # 结构化识别
            result = table_engine(img)

            # 当前页的输出子目录（仅用于文本结构保存）
            page_name = os.path.splitext(image_name)[0]
            page_output_dir = os.path.join(output_dir, page_name)
            os.makedirs(page_output_dir, exist_ok=True)

            # ✅ 保存结构化文本结果（去除图像）
            txt_path = os.path.join(page_output_dir, "res_0.txt")
            with codecs.open(txt_path, "w", encoding="utf-8") as f_txt:
                for region in result:
                    region_copy = region.copy()
                    region_copy.pop('img', None)
                    f_txt.write(json.dumps(region_copy, ensure_ascii=False) + "\n")

            # ✅ 保存结构可视化图（集中保存）
            im_show = draw_structure_result(pil_image, result, font_path=font_path)
            im_show = Image.fromarray(im_show)
            vis_path = os.path.join(vis_output_dir, f"{page_name}_vis.png")
            im_show.save(vis_path)

            # ✅ 保存图像类型区域（集中保存）
            for idx, block in enumerate(result):
                if block['type'] == 'figure' and 'img' in block:
                    figure_img = Image.fromarray(block['img'])
                    figure_path = os.path.join(figure_output_dir, f"{page_name}_figure_{idx + 1}.png")
                    figure_img.save(figure_path)

            # ✅ 保存表格类型区域（集中保存）
            for idx, block in enumerate(result):
                if block['type'] == 'table' and 'img' in block:
                    table_img = Image.fromarray(block['img'])
                    table_path = os.path.join(table_output_dir, f"{page_name}_table_{idx + 1}.png")
                    table_img.save(table_path)

            # ✅ 汇总并保存所有文本类型的内容（包括标题文本）
            text_blocks = [block for block in result if block['type'] in ['text', 'title']]  # 包括 title 类型
            text_lines = []

            # 对文本块按照 top 坐标进行排序
            sorted_text_blocks = sorted(text_blocks, key=lambda x: x['bbox'][1])  # 按照 y 坐标（top）进行排序

            for block in sorted_text_blocks:
                if 'res' in block:
                    lines = [item['text'] for item in block['res'] if 'text' in item]
                    text_lines.extend(lines)

            if text_lines:
                text_path = os.path.join(text_output_dir, f"{page_name}.txt")
                with open(text_path, 'w', encoding='utf-8') as f_text:
                    f_text.write("\n".join(text_lines))

            print(f"✅ 保存至: {page_output_dir}")

        except Exception as e:
            print(f"❌ 处理失败: {image_path}\n错误信息: {e}")
