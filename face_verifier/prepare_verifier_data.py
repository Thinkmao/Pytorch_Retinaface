import argparse
from pathlib import Path
from typing import List, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare binary verifier dataset from WIDER-style label.txt")
    parser.add_argument("--label_txt", type=str, required=True, help="WIDER-style label file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--min_size", type=int, default=10)
    parser.add_argument("--bg_per_image", type=int, default=2, help="random background crops per image")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_wider_label(label_txt: str) -> List[Tuple[str, List[List[float]]]]:
    rows: List[Tuple[str, List[List[float]]]] = []
    cur_path = None
    cur_labels: List[List[float]] = []

    with open(label_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if cur_path is not None:
                    rows.append((cur_path, cur_labels))
                cur_path = line[2:]
                cur_labels = []
            else:
                cur_labels.append([float(x) for x in line.split()])

    if cur_path is not None:
        rows.append((cur_path, cur_labels))
    return rows


def expand_box(x: int, y: int, w: int, h: int, margin: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * (1.0 + margin)
    nh = h * (1.0 + margin)

    x1 = max(0, int(round(cx - nw / 2.0)))
    y1 = max(0, int(round(cy - nh / 2.0)))
    x2 = min(img_w - 1, int(round(cx + nw / 2.0)))
    y2 = min(img_h - 1, int(round(cy + nh / 2.0)))
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()

    label_txt = Path(args.label_txt)
    img_root = label_txt.parent / "images"

    out_dir = Path(args.output_dir)
    pos_dir = out_dir / "face"
    neg_dir = out_dir / "non_face"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_wider_label(str(label_txt))

    list_path = out_dir / "train_list.txt"
    pos_id = 0
    neg_id = 0

    with list_path.open("w", encoding="utf-8") as fw:
        for rel_path, labels in rows:
            img_path = img_root / rel_path
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Positive crops from GT face boxes
            for label in labels:
                x, y, bw, bh = map(int, label[:4])
                if bw < args.min_size or bh < args.min_size:
                    continue
                x1, y1, x2, y2 = expand_box(x, y, bw, bh, args.margin, w, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                out_name = f"face/{pos_id:08d}.jpg"
                cv2.imwrite(str(out_dir / out_name), crop)
                fw.write(f"{out_name} 1\n")
                pos_id += 1

            # Simple background negatives from corners/center-like regions
            if args.bg_per_image > 0:
                candidates = [
                    (0, 0, w // 4, h // 4),
                    (w * 3 // 4, 0, w // 4, h // 4),
                    (0, h * 3 // 4, w // 4, h // 4),
                    (w * 3 // 4, h * 3 // 4, w // 4, h // 4),
                    (w // 3, h // 3, w // 3, h // 3),
                ]
                for i in range(min(args.bg_per_image, len(candidates))):
                    x, y, bw, bh = candidates[i]
                    if bw < args.min_size or bh < args.min_size:
                        continue
                    x1, y1, x2, y2 = expand_box(x, y, bw, bh, 0.0, w, h)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    out_name = f"non_face/{neg_id:08d}.jpg"
                    cv2.imwrite(str(out_dir / out_name), crop)
                    fw.write(f"{out_name} 0\n")
                    neg_id += 1

    print(f"Done. positives={pos_id}, negatives={neg_id}, list={list_path}")


if __name__ == "__main__":
    main()
