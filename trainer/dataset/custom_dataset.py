import json
from torch.utils.data import Dataset

from qwen_vl_utils import process_vision_info

class DocParseDataset(Dataset):
    """
    Qwen2-VL Document Parsing Dataset: Efficient loading for multimodal (vision+text) tasks.
    """
    def __init__(self, data_path, processor):
        self.data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                # Pre-read file line-by-line to minimize RAM footprint
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        self.data.append(json.loads(stripped))
        except FileNotFoundError:
            raise RuntimeError(f"Error: Data file not found at {data_path}")
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        prompt = item["prompt"]
        answer = item["answer"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ]
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        # Efficient prompt creation
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        # Remove batch dimension and set labels for Causal LM
        for k, v in inputs.items():
            if hasattr(v, "squeeze"):
                inputs[k] = v.squeeze(0)
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs