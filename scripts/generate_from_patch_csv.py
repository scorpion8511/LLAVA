import argparse
import os
from typing import Optional

import pandas as pd
import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_image(image_file: str) -> Image.Image:
    """Load an image from disk and convert it to RGB."""

    with Image.open(image_file) as img:
        return img.convert("RGB")


def build_prompt(
    question: str,
    tokenizer,
    conv_mode: str,
    mm_use_im_start_end: bool,
    use_mpt_roles: bool,
    device: torch.device,
) -> tuple[torch.Tensor, KeywordsStoppingCriteria]:
    """Prepare the prompt, token ids and stopping criteria."""

    conv = conv_templates[conv_mode].copy()

    if use_mpt_roles:
        roles = ("user", "assistant")
    else:
        roles = conv.roles

    if mm_use_im_start_end:
        question = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + question
        )
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv.append_message(roles[0], question)
    conv.append_message(roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    return input_ids, stopping_criteria


def generate_answer(
    model,
    tokenizer,
    question: str,
    image: Image.Image,
    image_processor,
    args,
    device: torch.device,
    dtype: torch.dtype,
    use_mpt_roles: bool,
) -> str:
    """Generate a response from the model for a given image/question pair."""

    image_tensor = process_images([image], image_processor, args)

    if isinstance(image_tensor, list):
        image_tensor = [img.to(device, dtype=dtype) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(device, dtype=dtype)

    input_ids, stopping_criteria = build_prompt(
        question,
        tokenizer,
        args.conv_mode,
        model.config.mm_use_im_start_end,
        use_mpt_roles,
        device,
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    return outputs


def prepare_question(row: pd.Series, args: argparse.Namespace) -> Optional[str]:
    if args.question_column:
        value = row.get(args.question_column, None)
        if pd.isna(value):
            return None
        return str(value)
    return args.question


def main():
    parser = argparse.ArgumentParser(description="Generate text for patches listed in a CSV file.")
    parser.add_argument("--csv-input", required=True, help="Path to the input CSV file containing patch information.")
    parser.add_argument(
        "--csv-output",
        required=True,
        help="Path where the output CSV with generated text should be saved.",
    )
    parser.add_argument(
        "--image-column",
        default="patch_path",
        help="Column name in the CSV that contains the path to the image patch.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional root directory to prepend to image paths read from the CSV.",
    )
    parser.add_argument(
        "--question",
        default="Describe the pathology patch in detail.",
        help="Default question to ask the model when generating descriptions.",
    )
    parser.add_argument(
        "--question-column",
        default=None,
        help="Column name containing per-row questions. Overrides --question when provided.",
    )
    parser.add_argument(
        "--output-column",
        default="generated_text",
        help="Name of the column where generated text will be stored in the output CSV.",
    )

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")

    args = parser.parse_args()

    disable_torch_init()

    if not os.path.exists(args.csv_input):
        raise FileNotFoundError(f"Input CSV not found: {args.csv_input}")

    df = pd.read_csv(args.csv_input)

    if args.image_column not in df.columns:
        raise ValueError(
            f"Column '{args.image_column}' not found in CSV columns: {', '.join(df.columns)}"
        )

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )

    if args.conv_mode is None:
        if "llama-2" in model_name.lower():
            args.conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            args.conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            args.conv_mode = "mpt"
        else:
            args.conv_mode = "llava_v0"

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(args.device)

    dtype = torch.float16 if model_device.type == "cuda" else torch.float32
    use_mpt_roles = args.conv_mode == "mpt"

    generated_texts = []

    for idx, row in df.iterrows():
        image_rel_path = row[args.image_column]
        if pd.isna(image_rel_path):
            generated_texts.append(None)
            continue

        image_path = str(image_rel_path)
        if args.image_root:
            image_path = os.path.join(args.image_root, image_path)

        if not os.path.exists(image_path):
            print(f"[Warning] Image not found for row {idx}: {image_path}")
            generated_texts.append(None)
            continue

        question = prepare_question(row, args)
        if question is None:
            generated_texts.append(None)
            continue

        image = load_image(image_path)
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            image=image,
            image_processor=image_processor,
            args=args,
            device=model_device,
            dtype=dtype,
            use_mpt_roles=use_mpt_roles,
        )
        generated_texts.append(answer)

    df[args.output_column] = generated_texts
    output_dir = os.path.dirname(args.csv_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(args.csv_output, index=False)


if __name__ == "__main__":
    main()
