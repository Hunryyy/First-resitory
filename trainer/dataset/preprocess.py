import json
import os
import logging
from typing import List
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def format_bbox(bbox: List[int]) -> str:
    return " ".join(str(int(x)) for x in bbox) if bbox and len(bbox) == 4 else "0 0 0 0"

def clean_text(text: str) -> str:
    return text.strip() if text else ""

def build_element_html(label: str, bbox_str: str, content: str) -> str:
    if label == "title":
        return f"<h2>{content}</h2>"
    if label == "paragraph":
        return f"<p>{content}</p>"
    if label == "table":
        return f'<div class="table" bbox="{bbox_str}"><table>{content}</table></div>'
    if label == "formula":
        return f'$${content}$$'
    if label in {"figure", "image"}:
        return f'<figure bbox="{bbox_str}"><img src="{content}"></figure>'
    return f'<div class="{label}" bbox="{bbox_str}">{content}</div>'

def convert_dataset(input_file: str, output_file: str, image_root_prefix: str = "data/images/"):
    valid_count = error_count = 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_index, line in enumerate(f_in):
            if not line.strip(): continue
            try:
                raw_item = json.loads(line)
                prompt_text = raw_item.get('prompt', "Parse the document layout and content into HTML format.")
                html_segments = [
                    build_element_html(
                        elem.get('label', 'paragraph'),
                        format_bbox(elem.get('bbox', [0, 0, 0, 0])),
                        clean_text(elem.get('content', ''))
                    )
                    for elem in raw_item.get('structure', [])
                ]
                final_html_answer = "<body>" + "".join(html_segments) + "</body>"
                image_path = os.path.join(image_root_prefix, raw_item.get('image'))
                output_item = {"image": image_path, "prompt": prompt_text, "answer": final_html_answer}
                f_out.write(json.dumps(output_item, ensure_ascii=False)+'\n')
                valid_count += 1
            except Exception:
                error_count += 1
    logging.error(f"成功: {valid_count}, 失败/跳过: {error_count}; 输出文件: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document parsing data preprocessor.")
    parser.add_argument("--input_path", type=str, default="data/train_raw.jsonl")
    parser.add_argument("--output_path", type=str, default="data/train_formatted.jsonl")
    parser.add_argument("--image_root", type=str, default="data/images/")
    args = parser.parse_args()
    convert_dataset(args.input_path, args.output_path, args.image_root)