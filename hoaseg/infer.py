from __future__ import annotations

import cv2
import numpy as np
import torch

from .model import UNet, predict_probs


def load_model(path: str, device: str) -> torch.nn.Module:
    ck = torch.load(path, map_location=device)
    model = UNet(base=ck.get("base", 32)).to(device)
    model.load_state_dict(ck["state"])
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


@torch.no_grad()
def predict_probs_batched(models: list[torch.nn.Module], imgs: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[i : i + batch]).to(device)
        p = None
        for m in models:
            q = predict_probs(m, x)
            p = q if p is None else p + q
        out.append((p / len(models)).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def cleanup(label: np.ndarray) -> np.ndarray:
    out = np.zeros_like(label)
    for cls in (1, 2):
        m = (label == cls).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == k).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(m)
        cv2.drawContours(filled, cnts, -1, 1, -1)
        out[(filled > 0) & (out == 0)] = cls
    return out


def probs_to_labels(probs: np.ndarray) -> np.ndarray:
    lab = probs.argmax(1).astype(np.uint8)
    return np.stack([cleanup(l) for l in lab])


def segment(models: list[torch.nn.Module], imgs: np.ndarray, device: str) -> np.ndarray:
    return probs_to_labels(predict_probs_batched(models, imgs, device))
