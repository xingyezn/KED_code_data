import json
from openai import OpenAI


def extract_entities(api_key, input_file, output_file, base_url="https://api.xiaoai.plus/v1"):
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt_template = '''
    {your_prompt}
    '''

    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 创建一个空列表，用于存储结果
    output_data = []

    # 遍历每个数据项
    for item in data:
        text = item['sentText']
        prompt = prompt_template.format(text=text)

        # 调用 OpenAI API 生成结果，使用 GPT-4 模型
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=2048,
            temperature=0.8,
            stream=False,
        )

        # 解析并获取生成内容
        extracted_entities = completion.choices[0].message.content.strip()

        try:
            print(f"Extracted entities: {extracted_entities}")

            if extracted_entities:
                entities_list = json.loads(extracted_entities)
            else:
                entities_list = None
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e.msg} at line {e.lineno} column {e.colno}")
            entities_list = extracted_entities

        new_item = {
            "sentText": text,
            "preREMentions": entities_list
        }
        output_data.append(new_item)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=4)

    print(f" {output_file} done.")


if __name__ == "__main__":
    # 隐藏 API 密钥
    api_key = "your_api_key_here"
    input_file = 'your_file_here'
    output_file = 'output_file'

    extract_entities(api_key, input_file, output_file)
