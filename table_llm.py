import os
import json
import traceback
import yaml
import re
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI
from prompt.table_prompt import TABLE_PROMPT_WITH_CUTLINE, TABLE_PROMPT_DEFAULT

# === 加载配置 ===
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
qwen_cfg = config["QwenVL"]
MAX_RETRY = qwen_cfg.get("max_retry", 3)

# === 初始化大模型客户端 ===
client = OpenAI(
    api_key=qwen_cfg["api_key"],
    base_url=qwen_cfg["base_url"]
)

def pil_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def remove_invalid_unicode_escapes(text: str) -> str:
    return re.sub(r'\\u[0-9a-fA-F]{0,3}[^0-9a-fA-F]', '', text)

def clean_json_output(raw_text: str) -> dict:
    try:
        raw_text = raw_text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json"):].strip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()
        raw_text = remove_invalid_unicode_escapes(raw_text)
        return json.loads(raw_text)
    except Exception as e:
        print(f"⚠️ JSON 格式化失败，返回原始字符串。错误: {e}")
        print(raw_text)
        raise e  # ❗抛出异常供上层重试判断

def call_qwen_vl(image: Image.Image, prompt: str, show_prompt: bool) -> str:
    try:
        img_b64 = pil_to_base64(image)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
        if show_prompt:
            print(f"Prompt:\n{prompt}\n")
        response = client.chat.completions.create(
            model=qwen_cfg.get("model", "qwen-vl-plus"),
            messages=[{"role": "user", "content": content}],
            extra_body={"safe_mode": qwen_cfg.get("safe_mode", True)}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Qwen-VL 调用失败：")
        traceback.print_exc()
        raise e

def run_table_extraction(pdf_name: str):
    ocr_dir = os.path.join("output", pdf_name, "ocr")
    table_dir = os.path.join(ocr_dir, "table")
    table_data_dir = os.path.join("output", pdf_name, "table_data")
    os.makedirs(table_data_dir, exist_ok=True)

    page_map = {}
    for page_folder in os.listdir(ocr_dir):
        page_path = os.path.join(ocr_dir, page_folder)
        if not os.path.isdir(page_path):
            continue
        res_path = os.path.join(page_path, "res_0.txt")
        if not os.path.exists(res_path):
            continue
        cutlines, tables = [], []
        with open(res_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "cutline":
                        cutlines.append({"text": " ".join(r["text"] for r in obj["res"]), "bbox": obj.get("bbox", [])})
                    elif obj.get("type") == "table" and "bbox" in obj:
                        tables.append({"bbox": obj["bbox"]})
                except json.JSONDecodeError:
                    continue
        page_map[page_folder] = {"cutlines": cutlines, "tables": tables}

    table_img_counter = {}

    for fname in sorted(os.listdir(table_dir)):
        if not fname.lower().endswith(".png"):
            continue
        img_path = os.path.join(table_dir, fname)
        pil_img = Image.open(img_path).convert("RGB")
        parts = fname.split("_")
        page_name = "_".join(parts[:3]) if len(parts) >= 5 else parts[0]

        count = table_img_counter.get(page_name, 0)
        table_img_counter[page_name] = count + 1

        page_info = page_map.get(page_name, {"cutlines": [], "tables": []})
        tables = page_info.get("tables", [])
        table_top = tables[count]["bbox"][1] if count < len(tables) else None

        best_cutline = None
        if table_top is not None:
            min_diff = float('inf')
            for cl in page_info.get("cutlines", []):
                if cl.get("bbox") and len(cl["bbox"]) >= 4:
                    cl_bottom = cl["bbox"][3]
                    if cl_bottom <= table_top:
                        diff = table_top - cl_bottom
                        if diff < min_diff:
                            min_diff = diff
                            best_cutline = cl["text"]
            if best_cutline is None and page_info["cutlines"]:
                for cl in page_info["cutlines"]:
                    if cl.get("bbox") and len(cl["bbox"]) >= 4:
                        diff = abs(cl["bbox"][3] - table_top)
                        if diff < min_diff:
                            min_diff = diff
                            best_cutline = cl["text"]

        use_cutline = qwen_cfg.get("enable_cutline_prompt", False)
        prompt = (
            TABLE_PROMPT_WITH_CUTLINE.format(cutline=best_cutline)
            if use_cutline and best_cutline
            else TABLE_PROMPT_DEFAULT
        )

        for attempt in range(1, MAX_RETRY + 1):
            try:
                response_text = call_qwen_vl(pil_img, prompt, show_prompt=use_cutline)
                json_output = clean_json_output(response_text)
                output_path = os.path.join(table_data_dir, fname.replace(".png", ".json"))
                with open(output_path, "w", encoding="utf-8") as f_out:
                    json.dump(json_output, f_out, ensure_ascii=False, indent=2)
                print(f"✅ 表格处理完成: {fname}")
                break
            except Exception as e:
                print(f"⚠️ 第 {attempt} 次解析失败: {fname}, 错误: {e}")
                if attempt == MAX_RETRY:
                    print(f"❌ 超出最大重试次数，跳过: {fname}")
