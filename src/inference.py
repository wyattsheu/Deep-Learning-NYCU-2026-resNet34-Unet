import argparse
import csv
import os

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.resnet34_unet import ResNet34_UNet
from models.unet import UNet
from utils import (
    calculate_dice_score,
    visualize_predictions_grid,
)


INPUT_SIZE = (572, 572)
TARGET_SIZE = (388, 388)


def center_crop_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = mask.shape
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return mask[top : top + target_h, left : left + target_w]


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask to RLE using column-major (Fortran) order."""
    pixels = mask.astype(np.uint8).flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def load_image_ids(data_dir: str, split_file: str = None, model_type: str = "UNet"):
    """Load image IDs from split file (first token per line)."""
    candidates = []
    if split_file:
        candidates.append(split_file)

    if model_type == "UNet":
        model_default_splits = [
            os.path.join(os.path.dirname(data_dir), "test_unet.txt"),
            os.path.join(os.path.dirname(data_dir), "test_res_unet.txt"),
        ]
    else:
        model_default_splits = [
            os.path.join(os.path.dirname(data_dir), "test_res_unet.txt"),
            os.path.join(os.path.dirname(data_dir), "test_unet.txt"),
        ]

    candidates.extend(model_default_splits)

    chosen_path = None
    for path in candidates:
        if os.path.exists(path):
            chosen_path = path
            break

    if chosen_path is None:
        raise FileNotFoundError(
            "Cannot find test split file. Checked: " + ", ".join(candidates)
        )

    image_ids = []
    with open(chosen_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_ids.append(line.split()[0])

    if len(image_ids) == 0:
        raise ValueError(f"No image IDs found in split file: {chosen_path}")

    return image_ids, chosen_path


class OxfordPetInferenceDataset(Dataset):
    def __init__(self, data_dir: str, image_ids, load_gt: bool):
        self.data_dir = os.path.abspath(data_dir)
        self.image_ids = image_ids
        self.load_gt = load_gt
        self.transform = T.Compose([T.Resize(INPUT_SIZE), T.ToTensor()])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_path = os.path.join(self.data_dir, "images", image_id + ".jpg")
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        image_tensor = self.transform(image)

        if not self.load_gt:
            return image_tensor, image_id, torch.tensor([orig_h, orig_w])

        mask_path = os.path.join(
            self.data_dir, "annotations", "trimaps", image_id + ".png"
        )
        mask = Image.open(mask_path).resize(INPUT_SIZE, resample=Image.NEAREST)
        mask_array = np.array(mask)
        binary_mask = np.zeros_like(mask_array, dtype=np.float32)
        binary_mask[mask_array == 1] = 1.0
        binary_mask = center_crop_mask(binary_mask, TARGET_SIZE[0], TARGET_SIZE[1])
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)
        return image_tensor, image_id, mask_tensor, torch.tensor([orig_h, orig_w])


class OxfordPetInferenceHFDataset(Dataset):
    def __init__(self, hf_dataset_name: str, split: str):
        self.dataset = load_dataset(hf_dataset_name, split=split)
        self.transform = T.Compose([T.Resize(INPUT_SIZE), T.ToTensor()])

        self.image_ids = []
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            image_id = item.get("filename")
            if image_id is None:
                image_id = item.get("image_id")
            if image_id is None:
                image_id = str(idx)
            self.image_ids.append(image_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_id = self.image_ids[idx]

        image = item["image"].convert("RGB")
        orig_w, orig_h = image.size
        image_tensor = self.transform(image)
        return image_tensor, image_id, torch.tensor([orig_h, orig_w])


def verify_unet_output_shape(model, device):
    with torch.no_grad():
        dummy = torch.zeros(1, 3, INPUT_SIZE[0], INPUT_SIZE[1], device=device)
        out = model(dummy)
    expected = TARGET_SIZE
    actual = tuple(out.shape[-2:])
    if actual != expected:
        raise ValueError(
            f"UNet output shape mismatch. Expected {expected}, got {actual}."
        )


def validate_submission_rows(rows, expected_ids):
    """Validate basic Kaggle-format constraints and return issue list."""
    issues = []

    row_ids = [row[0] for row in rows]
    expected_set = set(expected_ids)
    row_set = set(row_ids)

    missing = sorted(expected_set - row_set)
    extra = sorted(row_set - expected_set)

    if missing:
        issues.append(f"Missing image_ids: {len(missing)}")
    if extra:
        issues.append(f"Unknown image_ids: {len(extra)}")
    if len(row_ids) != len(row_set):
        issues.append("Duplicated image_id found in submission")

    for image_id, encoded_mask in rows:
        if not image_id:
            issues.append("Found empty image_id")
            break
        if encoded_mask and any(ch not in "0123456789 " for ch in encoded_mask):
            issues.append(f"Invalid RLE format detected for image_id={image_id}")
            break

    return issues


def run_inference(args):
    model_type = args.model_type
    model_path = args.model_path or f"saved_models/best_{model_type}.pth"
    data_dir = args.data_dir
    hf_dataset_name = args.hf_dataset_name
    hf_split = args.hf_split
    batch_size = args.batch_size
    submission_path = args.submission_path
    vis_dir = args.vis_dir
    threshold = args.threshold
    num_vis = args.num_vis

    os.makedirs(os.path.dirname(submission_path) or ".", exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if model_type == "UNet":
        model = UNet().to(device)
    else:
        model = ResNet34_UNet().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. Please train first."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if model_type == "UNet":
        verify_unet_output_shape(model, device)

    if hf_dataset_name:
        # HF 資料集雖有 mask，但 inference 階段只需讀 image 與檔名。
        gt_available = False
        test_dataset = OxfordPetInferenceHFDataset(
            hf_dataset_name=hf_dataset_name,
            split=hf_split,
        )
        image_ids = test_dataset.image_ids
        split_path = f"hf://{hf_dataset_name}/{hf_split}"
    else:
        image_ids, split_path = load_image_ids(
            data_dir=data_dir,
            split_file=args.split_file,
            model_type=model_type,
        )

        trimap_dir = os.path.join(data_dir, "annotations", "trimaps")
        gt_available = all(
            os.path.exists(os.path.join(trimap_dir, image_id + ".png"))
            for image_id in image_ids
        )

        test_dataset = OxfordPetInferenceDataset(
            data_dir=data_dir,
            image_ids=image_ids,
            load_gt=gt_available,
        )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    submissions = []
    total_dice = 0.0
    total_count = 0
    vis_buffer = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if gt_available:
                images, batch_image_ids, gt_masks, orig_sizes = batch
                gt_masks = gt_masks.to(device)
            else:
                images, batch_image_ids, orig_sizes = batch
                gt_masks = None

            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            if gt_available:
                batch_dice = calculate_dice_score(preds, gt_masks)
                batch_size_actual = images.size(0)
                total_dice += batch_dice * batch_size_actual
                total_count += batch_size_actual

            preds_np = preds.to(torch.uint8).cpu().numpy()

            for idx, (pred, image_id) in enumerate(zip(preds_np, batch_image_ids)):
                binary_mask = pred.squeeze(0).astype(np.uint8)

                # Kaggle evaluates masks in each test image's original resolution.
                orig_h = int(orig_sizes[idx, 0].item())
                orig_w = int(orig_sizes[idx, 1].item())
                mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((orig_w, orig_h), resample=Image.NEAREST)
                binary_mask_for_submit = (np.array(mask_img) > 127).astype(np.uint8)

                encoded_mask = rle_encode(binary_mask_for_submit)
                submissions.append((image_id, encoded_mask))

                if len(vis_buffer) < num_vis:
                    image_vis = images[idx].detach().cpu()
                    pred_vis = torch.from_numpy(binary_mask)
                    target_vis = None
                    if gt_available:
                        target_vis = gt_masks[idx].detach().cpu().squeeze(0)

                    vis_buffer.append((image_id, image_vis, pred_vis, target_vis))

    issues = validate_submission_rows(submissions, image_ids)

    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(submissions)

    print("=" * 60)
    print("Inference complete")
    print(f"Model type: {model_type}")
    print(f"Model checkpoint: {model_path}")
    print(f"Test split file: {split_path}")
    print(f"Total test images: {len(image_ids)}")
    print(f"Submission saved to: {submission_path}")
    print(f"Visualization samples showed: {len(vis_buffer)}")
    if model_type == "UNet":
        print(f"UNet shape check: input {INPUT_SIZE} -> output {TARGET_SIZE} (PASSED)")

    if issues:
        print("Kaggle format check: FAILED")
        for issue in issues:
            print(f" - {issue}")
    else:
        print("Kaggle format check: PASSED")

    if gt_available and total_count > 0:
        mean_dice = total_dice / total_count
        print(f"Simulated Kaggle Dice score (test-set average): {mean_dice:.6f}")
    else:
        print("Simulated Kaggle Dice score: skipped (ground-truth masks unavailable)")
    print("=" * 60)

    if vis_buffer:
        visualize_predictions_grid(vis_buffer)


def build_argparser():
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet inference for Kaggle")
    parser.add_argument(
        "--model-type",
        type=str,
        default="UNet",
        choices=["UNet", "ResNet34_UNet"],
        help="Model architecture to run",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Checkpoint path (.pth). If empty, use saved_models/best_<model_type>.pth",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/oxford-iiit-pet",
        help="Path to Oxford-IIIT Pet dataset root",
    )
    parser.add_argument(
        "--hf-dataset-name",
        type=str,
        default="",
        help="HF dataset repo name, e.g. user/oxford-pet-nycu-lab2. If set, use HF dataset instead of --data-dir.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="test_unet",
        help="HF split name to run inference on (train/val/test_unet/test_res_unet).",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default="",
        help="Optional custom test split file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary threshold for sigmoid outputs",
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default="submission.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="inference_outputs/vis_samples",
        help="Directory to save visualization examples",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=4,
        help="Number of test samples to visualize",
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    run_inference(parser.parse_args())
