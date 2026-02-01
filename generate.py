import argparse
import torch
from PIL import Image
from pathlib import Path
from pdf2image import convert_from_path

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor


def convert_pdf_to_images(pdf_path):
    """Convert PDF to PIL images (sync version)."""
    return convert_from_path(str(pdf_path))


def load_image_from_path(file_path):
    """
    Load an image from either an image file or PDF file.
    If PDF, only the first page is returned.
    
    Args:
        file_path: Path to image or PDF file
        
    Returns:
        PIL Image object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check if it's a PDF
    if file_path.suffix.lower() == '.pdf':
        print(f"Converting PDF to image (using first page only)...")
        images = convert_pdf_to_images(file_path)
        if not images:
            raise ValueError(f"Could not extract images from PDF: {file_path}")
        print(f"PDF has {len(images)} page(s), using page 1")
        return images[0]  # Return only the first page
    
    # Otherwise, treat as image
    else:
        return Image.open(file_path).convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from an image or PDF with nanoVLM")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default="shresht8/small-vlm1",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--input", type=str, default="test_data/1-2.pdf",
                        help="Path to input image or PDF file")
    parser.add_argument("--prompt", type=str, default="Extract all text in this image",
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=5,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of tokens per output")
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    template = f"Question: {args.prompt} Answer:"
    encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded["input_ids"].to(device)

    # Load image from either image file or PDF
    img = load_image_from_path(args.input)
    img_t = image_processor(img).unsqueeze(0).to(device)

    print("\nInput:\n ", args.prompt, "\n\nOutputs:")
    for i in range(args.generations):
        gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        print(f"  >> Generation {i+1}: {out}")


if __name__ == "__main__":
    main()