import torch, json, os, re
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

MERGED_PATH="output/mineru_merged_model"
ADAPTER_PATH="output/mineru_style_finetune"
BASE_MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
TEST_DATA_PATH="data/val_raw.jsonl"
SUBMIT_FILE="submit.jsonl"

def fix_html_structure(html_content):
    if not html_content: return ""
    html_content = html_content.strip()
    if not html_content.startswith("<body>"): html_content = "<body>"+html_content
    if not html_content.endswith("</body>"): html_content += "</body>"
    for tag in ["table", "div", "h2", "p"]:
        starts = len(re.findall(rf"<{tag}\b", html_content))
        ends = len(re.findall(rf"</{tag}>", html_content))
        html_content += f"</{tag}>" * max(0, starts-ends)
    if html_content.count("$$")%2>0: html_content += "$$"
    return html_content

@torch.inference_mode()
def inference():
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major>=8 else torch.float16
    is_merged = os.path.exists(os.path.join(MERGED_PATH,"config.json"))
    model_path = MERGED_PATH if is_merged else BASE_MODEL_PATH
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=dtype,attn_implementation="flash_attention_2",device_map="auto")
    processor = AutoProcessor.from_pretrained(MERGED_PATH if is_merged else ADAPTER_PATH)
    if not is_merged: model = PeftModel.from_pretrained(model,ADAPTER_PATH)
    model.eval()

    with open(TEST_DATA_PATH,'r',encoding='utf-8') as f_in, open(SUBMIT_FILE,'w',encoding='utf-8') as f_out:
        for line in tqdm(f_in,desc="Infer"):
            try:
                item = json.loads(line)
                img = item.get('image')
                prompt = item.get('prompt',"Parse the document layout and content into HTML format.")
                if not img or not os.path.exists(img): continue
                msg=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
                text = processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
                img_inputs,vid_inputs = process_vision_info(msg)
                inputs = processor(text=[text],images=img_inputs,videos=vid_inputs,padding=True,return_tensors="pt")
                inputs = {k: v.to(model.device) for k,v in inputs.items()}
                out_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=0.1)
                output_text = processor.batch_decode([oid[len(inputs["input_ids"][0]):] for oid in out_ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                res = {"image":os.path.basename(img),"prompt":prompt,"answer":fix_html_structure(output_text)}
                f_out.write(json.dumps(res,ensure_ascii=False)+'\n')
            except Exception: continue
    print(f"Inference Done: Results@{SUBMIT_FILE}")

if __name__=="__main__": inference()