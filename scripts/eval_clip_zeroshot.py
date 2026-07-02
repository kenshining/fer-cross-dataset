"""
CLIP ViT-B/32 零样本（zero-shot）跨数据集泛化评估。

CLIP 的标准使用方式：用文本 prompt 编码类别原型，图像编码器提取特征，
计算图像-文本余弦相似度进行分类，无需训练。

目的：回应编辑关于 vision-language / foundation-model baseline 的要求。
"""
from __future__ import annotations

import json, os, sys, tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

DATA_ROOT = Path("e:/scientific/小波/data")
RUNS_ROOT = _REPO / "runs" / "clip_baseline"
BATCH_SIZE = 32
NUM_CLASSES = 7

# 7 类表情的文本 prompt（CLIP 零样本标准做法）
EMOTION_PROMPTS = [
    "a photo of a person expressing anger",
    "a photo of a person expressing disgust",
    "a photo of a person expressing fear",
    "a photo of a person expressing happiness",
    "a photo of a person looking neutral",
    "a photo of a person expressing sadness",
    "a photo of a person expressing surprise",
]

os.makedirs(RUNS_ROOT, exist_ok=True)


def build_target_loader(dataset_name: str, batch_size: int, input_size: int = 224):
    from src.dataset_registry import REGISTRY
    from src.dataset_fer2013 import FER2013Dataset, fer2013_collate_fn

    if dataset_name == "rafdb":
        ds = REGISTRY["rafdb"]["dataset_cls"](DATA_ROOT / "RAF-DB", split="test")
        collate = None
    elif dataset_name == "fer2013":
        import csv
        csv_path = DATA_ROOT / "Fer2013" / "fer2013" / "fer2013.csv"
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write("emotion,pixels\n")
        with open(csv_path) as fin:
            next(fin)
            for line in fin:
                if "PublicTest" in line:
                    parts = line.strip().split(",", 2)
                    if len(parts) >= 2:
                        tmp.write(f"{parts[0]},{parts[1]}\n")
        tmp.close()
        ds = FER2013Dataset(Path(tmp.name))
        collate = fer2013_collate_fn
    elif dataset_name == "affectnet":
        ds = REGISTRY["affectnet"]["dataset_cls"](DATA_ROOT / "AffectNet", split="val")
        collate = REGISTRY["affectnet"]["collate_fn"]
    elif dataset_name == "ckplus":
        ds = REGISTRY["ckplus"]["dataset_cls"](DATA_ROOT / "CK+")
        collate = REGISTRY["ckplus"]["collate_fn"]
    elif dataset_name == "jaffe":
        ds = REGISTRY["jaffe"]["dataset_cls"](DATA_ROOT / "Jaffe")
        collate = REGISTRY["jaffe"]["collate_fn"]
    else:
        raise ValueError(f"Unknown target: {dataset_name}")

    t = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    class Wrapper(torch.utils.data.Dataset):
        def __init__(self, ds, t):
            self.ds = ds; self.t = t
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            pil, label = self.ds[idx]
            if not isinstance(pil, Image.Image):
                pil = pil.convert("RGB") if hasattr(pil, "convert") else Image.fromarray(np.array(pil))
            return self.t(pil), label
    return DataLoader(Wrapper(ds, t), batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate_zeroshot(model, processor, loader, text_features, device):
    """零样本评估：图像-文本余弦相似度分类。"""
    model.eval()
    correct = 0; total = 0
    tp = torch.zeros(NUM_CLASSES, device=device)
    fp = torch.zeros(NUM_CLASSES, device=device)
    fn = torch.zeros(NUM_CLASSES, device=device)

    for rgb, labels in loader:
        rgb, labels = rgb.to(device), labels.to(device)
        img_outputs = model.get_image_features(pixel_values=rgb)
        if hasattr(img_outputs, "pooler_output"):
            image_features = img_outputs.pooler_output
        else:
            image_features = img_outputs
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T) * model.logit_scale.exp()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        for c in range(NUM_CLASSES):
            tp[c] += ((pred == c) & (labels == c)).sum()
            fp[c] += ((pred == c) & (labels != c)).sum()
            fn[c] += ((pred != c) & (labels == c)).sum()
    acc = correct / max(total, 1)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"acc": acc, "macro_f1": f1.mean().item()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nCLIP ViT-B/32 Zero-Shot: cross-dataset evaluation")

    # 加载 CLIP 完整模型
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 编码文本 prompt 为类别原型
    text_inputs = processor(text=EMOTION_PROMPTS, return_tensors="pt", padding=True).to(device)
    text_outputs = model.get_text_features(**text_inputs)
    if hasattr(text_outputs, "pooler_output"):
        text_features = text_outputs.pooler_output
    else:
        text_features = text_outputs
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(f"  Text prototypes: {text_features.shape}")

    # 评估每个目标数据集
    targets = ["rafdb", "fer2013", "affectnet", "ckplus", "jaffe"]
    all_results = []

    for tgt in targets:
        try:
            loader = build_target_loader(tgt, BATCH_SIZE, 224)
            print(f"\n{tgt}: {len(loader.dataset)} test samples")
            m = evaluate_zeroshot(model, processor, loader, text_features, device)
            all_results.append({"target": tgt, "method": "CLIP-ZeroShot", **m})
            print(f"  Acc={m['acc']:.4f}, Macro-F1={m['macro_f1']:.4f}")
        except Exception as e:
            print(f"  [WARN] {tgt}: {e}")

    # 汇总
    print(f"\n{'='*55}")
    print(f"CLIP ViT-B/32 Zero-Shot — Cross-Dataset Macro-F1")
    print(f"{'Target':<12} {'Macro-F1':>10}")
    print("-" * 25)
    resnet_ref = {"rafdb": 0.559, "fer2013": 0.297, "affectnet": 0.249,
                  "ckplus": 0.174, "jaffe": 0.153}
    for r in all_results:
        tgt = r["target"]
        ref = resnet_ref.get(tgt, 0)
        print(f"{tgt:<12} {r['macro_f1']:>10.4f}  (ResNet-18: {ref:.3f})")

    out_path = RUNS_ROOT / "clip_zeroshot_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
