import argparse
import glob
import os

import cv2
import numpy as np
import torch

from hoaseg.infer import load_model, segment

SIZE = 180
UPPER = (236, 122, 58)
LOWER = (38, 128, 245)


def read_crop(path: str) -> tuple[np.ndarray, bool]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"{path} could not be read as an image")
    if img.shape == (SIZE, SIZE):
        return img, False
    h, w = img.shape
    side = min(h, w)
    y, x = (h - side) // 2, (w - side) // 2
    return cv2.resize(img[y : y + side, x : x + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA), True


def stretch(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    v = img[img > 10]
    if v.size < 50:
        return img
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if hi - lo < 5:
        return img
    out = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1) * 235 + 10
    out[img <= 10] = 0
    return out.astype(np.uint8)


def overlay(img: np.ndarray, lab: np.ndarray, scale: int = 4, alpha: float = 0.48) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(stretch(img), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR).astype(np.float32)
    for cls, col in ((1, UPPER), (2, LOWER)):
        m = cv2.resize((lab == cls).astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        soft = np.clip((m - 0.5) * 4 + 0.5, 0, 1)[..., None]
        rgb = rgb * (1 - alpha * soft) + np.array(col, np.float32) * alpha * soft
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Segment the upper and the lower bone of finger joints on X-ray crops.")
    ap.add_argument("images", nargs="+", help="X-ray crops (PNG or JPEG)")
    ap.add_argument("--out", default="segmentations", help="folder for the results (default: segmentations)")
    ap.add_argument("--checkpoints", default=os.path.join(here, "checkpoints"), help="folder with the fold models (*.pt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.checkpoints, "*.pt")))
    if not paths:
        raise SystemExit(f"no model files (*.pt) in {a.checkpoints}")
    models = [load_model(p, a.device) for p in paths]
    crops = [read_crop(p) for p in a.images]
    imgs = np.stack([c[0] for c in crops])
    labels = segment(models, imgs, a.device)

    os.makedirs(a.out, exist_ok=True)
    for path, (img, resized), lab in zip(a.images, crops, labels):
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(a.out, name + "_mask.png"), lab)
        cv2.imwrite(os.path.join(a.out, name + "_segmentation.png"), overlay(img, lab))
        note = " (resized to 180 x 180)" if resized else ""
        print(f"{name}: upper bone {int((lab == 1).sum())} px, lower bone {int((lab == 2).sum())} px{note}")
    print(f"{len(a.images)} image(s), {len(models)} models on {a.device} -> {a.out}/")


if __name__ == "__main__":
    main()
