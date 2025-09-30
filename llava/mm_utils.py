from PIL import Image
from io import BytesIO
import base64

import numpy as np
import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    if image_aspect_ratio == 'pad':
        processed_images = []
        for image in images:
            image = expand2square(
                image,
                tuple(int(x * 255) for x in image_processor.image_mean),
            )
            processed = image_processor(
                images=[image],
                return_tensors=None,
                padding=True,
            )["pixel_values"]

            if not isinstance(processed, (list, tuple)):
                processed = [processed]

            processed_images.extend(processed)

        if len(processed_images) == 1:
            return _ensure_batch_tensor(processed_images[0])

        tensor_list = [_ensure_tensor(img) for img in processed_images]

        if all(t.shape == tensor_list[0].shape for t in tensor_list):
            return torch.stack(tensor_list, dim=0)
        return tensor_list

    processed = image_processor(images=images, return_tensors=None)
    pixel_values = processed["pixel_values"]

    if isinstance(pixel_values, torch.Tensor):
        return pixel_values

    if not isinstance(pixel_values, (list, tuple)):
        pixel_values = [pixel_values]

    tensor_list = [_ensure_tensor(img) for img in pixel_values]

    if len(tensor_list) == 1:
        return tensor_list[0].unsqueeze(0)
    if all(t.shape == tensor_list[0].shape for t in tensor_list):
        return torch.stack(tensor_list, dim=0)
    return tensor_list


def _ensure_tensor(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like
    if isinstance(array_like, np.ndarray):
        # torch.from_numpy requires NumPy support in the PyTorch build, which may
        # be unavailable in runtime environments that pin to NumPy 1.x wheels.
        # Converting through Python lists avoids that dependency at the cost of
        # an extra copy but keeps inference functional.
        return torch.tensor(array_like.tolist(), dtype=torch.float32)
    if isinstance(array_like, list):
        return torch.tensor(array_like, dtype=torch.float32)
    if isinstance(array_like, (float, int)):
        return torch.tensor(array_like, dtype=torch.float32)
    if hasattr(array_like, "tolist"):
        return torch.tensor(array_like.tolist(), dtype=torch.float32)
    raise TypeError(f"Unsupported image type: {type(array_like)!r}")


def _ensure_batch_tensor(tensor):
    tensor = _ensure_tensor(tensor)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
