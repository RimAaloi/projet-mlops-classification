#!/usr/bin/env python3
"""
Utility script to extract pixel arrays from Fashion MNIST dataset or custom images for API testing.

Usage:
    # From Fashion MNIST CSV file
    python src/get_test_pixels.py <csv_file> [--index N] [--random] [--json]
    
    # From custom image file (PNG, JPG, etc.)
    python src/get_test_pixels.py <image_file> --image [--json]

Examples:
    # Get first sample from CSV
    python src/get_test_pixels.py data/fashion-mnist/fashion-mnist_test.csv

    # Get sample at specific index
    python src/get_test_pixels.py data/fashion-mnist/fashion-mnist_test.csv --index 42

    # Get random sample
    python src/get_test_pixels.py data/fashion-mnist/fashion-mnist_test.csv --random

    # Output as JSON (ready for API)
    python src/get_test_pixels.py data/fashion-mnist/fashion-mnist_test.csv --random --json
    
    # From custom image file
    python src/get_test_pixels.py my_shirt.png --image
    
    # From custom image with JSON output
    python src/get_test_pixels.py my_shirt.jpg --image --json --model cnn
"""

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Fashion MNIST class labels
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def load_sample(file_path: str, index: int = 0, random_sample: bool = False) -> tuple:
    """
    Load a sample from Fashion MNIST CSV file.
    
    Args:
        file_path: Path to the CSV file
        index: Index of the sample to extract (0-based)
        random_sample: If True, select a random sample
    
    Returns:
        Tuple of (label, pixels, index)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read file and skip header
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Remove header
    header = lines[0]
    data_lines = lines[1:]
    
    total_samples = len(data_lines)
    
    if random_sample:
        index = random.randint(0, total_samples - 1)
    elif index >= total_samples:
        raise IndexError(f"Index {index} out of range. File has {total_samples} samples (0-{total_samples-1})")
    
    # Parse the selected line
    line = data_lines[index].strip()
    values = line.split(',')
    
    label = int(values[0])
    # Normalize pixels to 0-1 range (original values are 0-255)
    pixels = [float(v) / 255.0 for v in values[1:]]
    
    return label, pixels, index


def load_image(file_path: str, invert: bool = True) -> list:
    """
    Load a custom image file and convert to 784 pixel array.
    
    The image will be:
    1. Converted to grayscale
    2. Resized to 28x28 pixels
    3. Normalized to 0-1 range
    4. Optionally inverted (Fashion MNIST has white items on black background)
    
    Args:
        file_path: Path to the image file (PNG, JPG, etc.)
        invert: If True, invert colors (default for Fashion MNIST style)
    
    Returns:
        List of 784 float values (0-1)
    """
    if not PIL_AVAILABLE:
        raise ImportError(
            "Pillow is required for image loading. Install with: pip install Pillow"
        )
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    
    # Open and convert to grayscale
    img = Image.open(path).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to pixel values
    pixels = list(img.getdata())
    
    # Normalize to 0-1 range
    pixels = [p / 255.0 for p in pixels]
    
    # Invert if needed (Fashion MNIST: white clothing on black background)
    if invert:
        pixels = [1.0 - p for p in pixels]
    
    return pixels


def visualize_ascii(pixels: list, width: int = 28) -> str:
    """Create ASCII art visualization of the image."""
    chars = " .:-=+*#%@"
    result = []
    
    for i in range(0, len(pixels), width):
        row = pixels[i:i + width]
        line = "".join(chars[min(int(p * 9), 9)] for p in row)
        result.append(line)
    
    return "\n".join(result)


def main():
    parser = argparse.ArgumentParser(
        description="Extract pixel arrays from Fashion MNIST or custom images for API testing"
    )
    parser.add_argument(
        "file_path",
        help="Path to Fashion MNIST CSV file or image file (PNG, JPG, etc.)"
    )
    parser.add_argument(
        "--image", "-img",
        action="store_true",
        help="Treat input as image file instead of CSV"
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Don't invert image colors (for images already on black background)"
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        default=0,
        help="Index of sample to extract from CSV (default: 0)"
    )
    parser.add_argument(
        "--random", "-r",
        action="store_true",
        help="Select a random sample from CSV"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON ready for API request"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["mlp", "cnn", "transfer"],
        default=None,
        help="Include model selection in JSON output"
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Skip ASCII visualization"
    )
    
    args = parser.parse_args()
    
    try:
        if args.image:
            # Load from image file
            pixels = load_image(args.file_path, invert=not args.no_invert)
            label = None
            index = None
            source_type = "image"
        else:
            # Load from CSV
            label, pixels, index = load_sample(
                args.file_path,
                index=args.index,
                random_sample=args.random
            )
            source_type = "csv"
        
        if args.json:
            # Output JSON ready for API
            request_body = {"data": [pixels]}
            if args.model:
                request_body["model"] = args.model
            print(json.dumps(request_body))
        else:
            # Human-readable output
            print(f"{'='*50}")
            if source_type == "csv":
                print(f"Source: CSV file")
                print(f"Sample Index: {index}")
                print(f"True Label: {label} ({CLASS_NAMES[label]})")
            else:
                print(f"Source: Image file")
                print(f"File: {args.file_path}")
                print(f"Colors inverted: {not args.no_invert}")
            print(f"Pixel Count: {len(pixels)}")
            print(f"{'='*50}")
            
            if not args.no_visual:
                print("\nASCII Preview:")
                print(visualize_ascii(pixels))
                print()
            
            print("\n📋 API Request Body (copy this):")
            print("-" * 50)
            request_body = {"data": [pixels]}
            if args.model:
                request_body["model"] = args.model
            print(json.dumps(request_body, indent=2)[:500] + "...")
            
            print("\n🧪 Quick Test Command:")
            print("-" * 50)
            if source_type == "csv":
                print(f"""curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '$(python3 src/get_test_pixels.py {args.file_path} -i {index} --json{" -m " + args.model if args.model else ""})'""")
                print(f"\n✅ Expected class: {CLASS_NAMES[label]}")
            else:
                invert_flag = "" if not args.no_invert else " --no-invert"
                print(f"""curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '$(python3 src/get_test_pixels.py {args.file_path} --image{invert_flag} --json{" -m " + args.model if args.model else ""})'""")
                print(f"\n🔮 Model will predict the class")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except IndexError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
