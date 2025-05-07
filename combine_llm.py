import os
import json
import traceback
import yaml
from openai import OpenAI
from prompt.combine_prompt import PROMPT  # 导入清洗提示

# === 加载配置 ===
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
text_llm_cfg = config["OPENAI"]  # 使用 TextLLM 配置
MAX_RETRY = text_llm_cfg.get("max_retry", 3)

# === 初始化大模型客户端 ===
client = OpenAI(
    api_key=text_llm_cfg["api_key"],
    base_url=text_llm_cfg["base_url"]
)


def call_openai(json_data: str, show_prompt: bool) -> str:
    try:
        content = [
            {"role": "system", "content": "你是一个电子器件数据手册的专家，帮助用户清洗器件名称和电气参数。"},
            {"role": "user", "content": PROMPT.format(json_data=json_data)}
        ]
        if show_prompt:
            print(f"Prompt:\n{PROMPT.format(json_data=json_data)}\n")

        response = client.chat.completions.create(
            model=text_llm_cfg.get("model", "deepseek-chat"),
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
        raw_text = raw_text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json"):].strip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()
        return json.loads(raw_text)
    except Exception as e:
        print(f"⚠️ JSON 格式化失败，返回原始字符串。错误: {e}")
        print(raw_text)
        return {"raw": raw_text}


def run_combine_llm(parameter_json_file: str):
    # 读取parameter.json文件
    try:
        with open(parameter_json_file, 'r', encoding='utf-8') as f:
            parameter_json_data = f.read()

        # 重试逻辑
        for attempt in range(1, MAX_RETRY + 1):
            try:
                response_text = call_openai(parameter_json_data, show_prompt=False)
                json_output = clean_json_output(response_text)

                # 保存清洗后的最终JSON
                final_parameter_path = os.path.join(os.path.dirname(parameter_json_file), "final_parameter.json")
                with open(final_parameter_path, 'w', encoding='utf-8') as f_out:
                    json.dump(json_output, f_out, ensure_ascii=False, indent=2)

                print(f"✅ 合并和清洗完成，结果已保存到: {final_parameter_path}")
                break  # 成功就退出重试
            except Exception as e:
                print(f"⚠️ 第 {attempt} 次合并失败: {parameter_json_file}, 错误: {e}")
                if attempt == MAX_RETRY:
                    print(f"❌ 超出最大重试次数，跳过: {parameter_json_file}")

    except Exception as e:
        print(f"❌ 处理失败: {parameter_json_file}, 错误: {e}")
