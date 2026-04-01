import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from face_verifier.model import FaceBinaryClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run binary verifier on face crops")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image path or a txt file that lists image paths")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_paths(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return [str(p)]
    paths = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(line)
    return paths


def preprocess(image_path: str, image_size: int) -> torch.Tensor:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FaceBinaryClassifier(backbone=args.backbone).to(device)
    ckpt = torch.load(args.model, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    paths = load_paths(args.input)
    with torch.no_grad():
        for path in paths:
            x = preprocess(path, args.image_size).to(device)
            logit = model(x)[0]
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= args.threshold else 0
            print(f"{path}\tprob={prob:.6f}\tpred={pred}")


if __name__ == "__main__":
    main()
