from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RetinaFace over a Hugging Face dataset for negative mining and export WIDERFace-style labels.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Hugging Face online examples:\n"
            "  --hf_dataset_id username/PASS --hf_split train\n"
            "  --hf_dataset_id username/PASS --hf_split train --hf_streaming\n"
        ),
    )
    parser.add_argument("--trained_model", required=True, type=str, help="Path to RetinaFace weights.")
    parser.add_argument("--network", default="resnet50", choices=["mobile0.25", "resnet50"], help="Backbone type.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Force CPU inference.")

    parser.add_argument("--confidence_threshold", default=0.3, type=float, help="Confidence filter before NMS.")
    parser.add_argument("--nms_threshold", default=0.4, type=float, help="NMS IoU threshold.")
    parser.add_argument("--top_k", default=5000, type=int, help="Keep top-k before NMS.")
    parser.add_argument("--keep_top_k", default=750, type=int, help="Keep top-k after NMS.")
    parser.add_argument("--vis_thres", default=0.6, type=float, help="Final score threshold for label export.")

    parser.add_argument(
        "--input_size",
        default="",
        type=str,
        help="Optional inference input size, e.g. 320x320. Empty means using original image size.",
    )

    parser.add_argument("--hf_dataset_id", required=True, type=str, help="Hugging Face dataset id, e.g. org/pass.")
    parser.add_argument("--hf_config", default=None, type=str, help="Optional Hugging Face dataset config/subset name.")
    parser.add_argument("--hf_split", default="train", type=str, help="Hugging Face split name.")
    parser.add_argument("--hf_image_column", default="image", type=str, help="Image column name in HF samples.")
    parser.add_argument(
        "--hf_path_column",
        default="path",
        type=str,
        help="Optional path/id column for output naming; fallback uses index if missing.",
    )
    parser.add_argument("--hf_streaming", action="store_true", default=False, help="Enable HF streaming mode.")
    parser.add_argument("--max_samples", default=0, type=int, help="Limit processed images; 0 means all.")

    parser.add_argument(
        "--output_label",
        default="./negative_mining_labels.txt",
        type=str,
        help="Output label file path in WIDERFace label.txt format.",
    )
    parser.add_argument(
        "--output_face_images",
        default="./negative_mining_face_images",
        type=str,
        help="Directory to save original images that contain detected faces.",
    )
    parser.add_argument(
        "--ir_gray",
        action="store_true",
        default=False,
        help="Convert each input image to IR-style grayscale before detection and output.",
    )
    return parser.parse_args()


def parse_size(size: str) -> Optional[Tuple[int, int]]:
    if not size:
        return None
    norm = size.lower().replace(" ", "")
    sep = "x" if "x" in norm else ","
    parts = norm.split(sep)
    if len(parts) != 2:
        raise ValueError("--input_size should be like 320x320 or 320,320")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError("--input_size width/height must be positive")
    return w, h


def check_keys(model: torch.nn.Module, pretrained_state_dict: dict) -> bool:
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print(f"Missing keys:{len(missing_keys)}")
    print(f"Unused checkpoint keys:{len(unused_pretrained_keys)}")
    print(f"Used keys:{len(used_pretrained_keys)}")
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict: dict, prefix: str) -> dict:
    print(f"remove prefix '{prefix}'")
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model: torch.nn.Module, pretrained_path: str, load_to_cpu: bool) -> torch.nn.Module:
    print(f"Loading pretrained model from {pretrained_path}")
    if load_to_cpu or not torch.cuda.is_available():
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def iter_hf_images(
    dataset_id: str,
    config_name: Optional[str],
    split: str,
    image_column: str,
    path_column: str,
    streaming: bool,
) -> Iterator[Tuple[str, np.ndarray]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("Please install Hugging Face datasets first: pip install datasets") from exc

    ds = load_dataset(dataset_id, name=config_name, split=split, streaming=streaming)
    print(f"Loaded Hugging Face dataset: {dataset_id} [{split}] (streaming={streaming})")

    for idx, sample in enumerate(ds):
        if image_column not in sample:
            raise RuntimeError(f"HF sample does not contain image column '{image_column}'.")

        pil_img = sample[image_column]
        img_rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        rel_name = sample.get(path_column)
        if not isinstance(rel_name, str) or not rel_name:
            rel_name = f"hf_{idx:08d}.jpg"
        else:
            rel_name = rel_name.replace('\\', '/')
            if not Path(rel_name).suffix:
                rel_name = f"{rel_name}.jpg"

        yield rel_name, img_bgr


def detect_faces(
    net: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    img_raw: np.ndarray,
    input_size: Optional[Tuple[int, int]],
    confidence_threshold: float,
    nms_threshold: float,
    top_k: int,
    keep_top_k: int,
) -> np.ndarray:
    img = np.float32(img_raw)
    orig_h, orig_w, _ = img.shape

    if input_size is not None:
        target_w, target_h = input_size
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    infer_h, infer_w, _ = img.shape

    scale = torch.tensor([infer_w, infer_h, infer_w, infer_h], device=device)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    loc, conf, landms = net(img)

    priorbox = PriorBox(cfg, image_size=(infer_h, infer_w))
    priors = priorbox.forward().to(device)
    prior_data = priors.data

    boxes = decode(loc.data.squeeze(0), prior_data, cfg["variance"])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg["variance"])
    scale1 = torch.tensor(
        [infer_w, infer_h, infer_w, infer_h, infer_w, infer_h, infer_w, infer_h, infer_w, infer_h],
        device=device,
    )
    landms = (landms * scale1).cpu().numpy()

    inds = np.where(scores > confidence_threshold)[0]
    boxes, landms, scores = boxes[inds], landms[inds], scores[inds]

    order = scores.argsort()[::-1][:top_k]
    boxes, landms, scores = boxes[order], landms[order], scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :][:keep_top_k, :]
    landms = landms[keep][:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    if input_size is not None and len(dets) > 0:
        sx = float(orig_w) / float(infer_w)
        sy = float(orig_h) / float(infer_h)
        dets[:, [0, 2]] *= sx
        dets[:, [1, 3]] *= sy
        dets[:, [5, 7, 9, 11, 13]] *= sx
        dets[:, [6, 8, 10, 12, 14]] *= sy

    return dets


def convert_to_ir_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Approximate IR camera style grayscale and keep 3 channels for RetinaFace input."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def to_label_line(box: Sequence[float], image_w: int, image_h: int) -> str:
    x1, y1, x2, y2 = box[:4]
    x1 = int(np.clip(np.floor(x1), 0, image_w - 1))
    y1 = int(np.clip(np.floor(y1), 0, image_h - 1))
    x2 = int(np.clip(np.ceil(x2), 0, image_w - 1))
    y2 = int(np.clip(np.ceil(y2), 0, image_h - 1))

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return f"{x1} {y1} {w} {h} 0 0 0 0 0 0"


def run_mining(
    image_iter: Iterable[Tuple[str, np.ndarray]],
    net: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    args: argparse.Namespace,
    input_size: Optional[Tuple[int, int]],
) -> None:
    label_path = Path(args.output_label)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    out_img_dir = Path(args.output_face_images)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    total_faces = 0
    seen_count = 0

    with label_path.open("w", encoding="utf-8") as fw:
        for rel_name, img_raw in image_iter:
            seen_count += 1
            if args.max_samples > 0 and seen_count > args.max_samples:
                break

            img_for_pipeline = convert_to_ir_gray(img_raw) if args.ir_gray else img_raw

            dets = detect_faces(
                net=net,
                cfg=cfg,
                device=device,
                img_raw=img_for_pipeline,
                input_size=input_size,
                confidence_threshold=args.confidence_threshold,
                nms_threshold=args.nms_threshold,
                top_k=args.top_k,
                keep_top_k=args.keep_top_k,
            )

            dets = np.array([b for b in dets if b[4] >= args.vis_thres], dtype=np.float32)
            if dets.size == 0:
                continue

            h, w = img_for_pipeline.shape[:2]
            fw.write(f"#{rel_name}\n")
            fw.write(f"{len(dets)}\n")
            for b in dets:
                fw.write(to_label_line(b, w, h) + "\n")

            dst_path = out_img_dir / rel_name
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_path), img_for_pipeline)
            valid_count += 1
            total_faces += len(dets)

            if seen_count % 100 == 0:
                print(f"Processed {seen_count} images, kept {valid_count} images.")

    print(
        f"Done. Seen images: {seen_count}, images with faces: {valid_count}, total faces: {total_faces}. "
        f"Label file: {label_path}, copied images dir: {out_img_dir}"
    )


def main() -> None:
    args = parse_args()
    input_size = parse_size(args.input_size)

    if not args.cpu and not torch.cuda.is_available():
        print("CUDA is not available, automatically switching to CPU inference.")
        args.cpu = True

    torch.set_grad_enabled(False)

    cfg = cfg_mnet if args.network == "mobile0.25" else cfg_re50
    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()

    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    cudnn.benchmark = True

    image_iter = iter_hf_images(
        dataset_id=args.hf_dataset_id,
        config_name=args.hf_config,
        split=args.hf_split,
        image_column=args.hf_image_column,
        path_column=args.hf_path_column,
        streaming=args.hf_streaming,
    )

    run_mining(image_iter=image_iter, net=net, cfg=cfg, device=device, args=args, input_size=input_size)


if __name__ == "__main__":
    main()
