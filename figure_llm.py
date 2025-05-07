import os
import yaml
import base64
import traceback
from PIL import Image
from io import BytesIO
from openai import OpenAI
from prompt.figure_prompt import FIGURE_PROMPT

# === 加载配置 ===
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
qwen_cfg = config["QwenVL"]

# === 初始化大模型客户端 ===
client = OpenAI(
    api_key=qwen_cfg["api_key"],
    base_url=qwen_cfg["base_url"]
)

def pil_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def call_qwen_figure(image: Image.Image, prompt: str) -> str:
    try:
        img_b64 = pil_to_base64(image)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
        response = client.chat.completions.create(
            model=qwen_cfg.get("model", "qwen-vl-plus"),
            messages=[{"role": "user", "content": content}],
            extra_body={"safe_mode": qwen_cfg.get("safe_mode", True)}
        )
        return response.choices[0].message.content.strip().lower()
    except Exception:
        print("❌ Qwen-VL 图像识别失败：")
        traceback.print_exc()
        return "false"

def run_figure_analysis(pdf_name: str):
    figure_dir = os.path.join("output", pdf_name, "ocr", "figure")
    save_dir = os.path.join("output", pdf_name, "parameter_curve")
    os.makedirs(save_dir, exist_ok=True)

    for fname in os.listdir(figure_dir):
        if fname.lower().endswith(".png"):
            fpath = os.path.join(figure_dir, fname)
            image = Image.open(fpath).convert("RGB")

            result = call_qwen_figure(image, FIGURE_PROMPT)
            if "true" in result:
                image.save(os.path.join(save_dir, fname))
                print(f"✅ 是电气参数图: {fname}")
            else:
                print(f"🚫 不是电气参数图: {fname}")
