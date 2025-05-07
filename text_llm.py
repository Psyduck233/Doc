import os
import json
import traceback
import yaml
from openai import OpenAI
from prompt.text_prompt import TEXT_PROMPT  # 导入文本提示

# === 加载配置 ===
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
qwen_cfg = config["OPENAI"]  # 这里使用新的配置项 `TextLLM`
MAX_RETRY = qwen_cfg.get("max_retry", 3)

# === 初始化大模型客户端 ===
client = OpenAI(
    api_key=qwen_cfg["api_key"],
    base_url=qwen_cfg["base_url"]
)

def call_openai(text: str, show_prompt: bool) -> str:
    try:
        content = [
            {"role": "system", "content": "你是一个电子器件数据手册的专家，帮助用户从文本中抽取器件名称和器件的电气参数。"},
            {"role": "user", "content": TEXT_PROMPT.format(text=text)}
        ]
        if show_prompt:
            print(f"Prompt:\n{TEXT_PROMPT.format(text=text)}\n")
        response = client.chat.completions.create(
            model=qwen_cfg.get("model", "deepseek-chat"),
            messages=content,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ OpenAI 调用失败：")
        traceback.print_exc()
        raise e

def clean_json_output(raw_text: str) -> dict:
    try:
        # 清洗返回的内容，确保返回有效的 JSON 格式
        raw_text = raw_text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json"):].strip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()
        return json.loads(raw_text)
    except Exception as e:
        print(f"⚠️ JSON 格式化失败，返回原始字符串。错误: {e}")
        print(raw_text)
        raise e

def run_text_extraction(pdf_name: str):
    ocr_dir = os.path.join("output", pdf_name, "ocr", "text")
    text_data_dir = os.path.join("output", pdf_name, "text_data")
    os.makedirs(text_data_dir, exist_ok=True)

    for txt_file in sorted(os.listdir(ocr_dir)):
        if txt_file.lower().endswith(".txt"):
            txt_path = os.path.join(ocr_dir, txt_file)
            page_name = os.path.splitext(txt_file)[0]

            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()

                # 重试逻辑
                for attempt in range(1, MAX_RETRY + 1):
                    try:
                        response_text = call_openai(text, show_prompt=False)
                        json_output = clean_json_output(response_text)
                        output_path = os.path.join(text_data_dir, f"{page_name}.json")
                        with open(output_path, "w", encoding="utf-8") as f_out:
                            json.dump(json_output, f_out, ensure_ascii=False, indent=2)
                        print(f"✅ 文本处理完成: {page_name}")
                        break  # 成功就退出重试
                    except Exception as e:
                        print(f"⚠️ 第 {attempt} 次解析失败: {page_name}, 错误: {e}")
                        if attempt == MAX_RETRY:
                            print(f"❌ 超出最大重试次数，跳过: {page_name}")

            except Exception as e:
                print(f"❌ 处理失败: {txt_file}, 错误: {e}")
