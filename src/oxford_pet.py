import os
import random


import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from datasets import load_dataset


class LetterBoxResize:
    def __init__(self, target_size, interpolation=InterpolationMode.BILINEAR, fill=0):
        self.target_size = target_size
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(h, w)

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = TF.resize(
            img,
            (new_h, new_w),
            interpolation=self.interpolation,
        )

        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top

        return TF.pad(
            resized,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


class OxfordPetDataset(Dataset):
    HF_DATASET_NAME = "wyattsheu/oxford-pet-full-raw"

    # 🌟 新增 image_size 與 mask_size，預設為 UNet 的尺寸
    def __init__(
        self,
        data_dir="dataset",
        split="train",
        image_size=572,
        mask_size=388,
        return_mask_for_test=False,
        return_unpadded_for_test=False,
    ):
        self.split = split
        self.data_dir = os.path.abspath(data_dir)
        self.image_size = image_size
        self.mask_size = mask_size
        self.return_mask_for_test = return_mask_for_test
        self.return_unpadded_for_test = return_unpadded_for_test

        txt_path = os.path.join(self.data_dir, f"{split}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Split file not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            self.target_names = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                self.target_names.append(line.split()[0])

        self.use_local = False
        self.local_root = None
        candidate_roots = [
            os.path.join(self.data_dir, "oxford-iiit-pet"),
            self.data_dir,
        ]
        for root in candidate_roots:
            images_dir = os.path.join(root, "images")
            trimaps_dir = os.path.join(root, "annotations", "trimaps")
            if os.path.isdir(images_dir) and os.path.isdir(trimaps_dir):
                self.use_local = True
                self.local_root = root
                break

        if self.use_local:
            print(f"使用本地資料夾: {self.local_root}")
        else:
            print(f"正在從雲端索引完整資料庫 ({split})...")
            raw_ds = load_dataset(self.HF_DATASET_NAME)

            if isinstance(raw_ds, dict):
                if "train" in raw_ds:
                    full_ds = raw_ds["train"]
                else:
                    first_split = next(iter(raw_ds.keys()))
                    full_ds = raw_ds[first_split]
            else:
                full_ds = raw_ds

            self.name_to_idx = {name: i for i, name in enumerate(full_ds["filename"])}
            self.ds = full_ds

        # 🌟 計算需要鏡像填充的像素大小 (例如 572 - 388 = 184，除以2 = 92)
        pad_size = (self.image_size - self.mask_size) // 2

        # 先做等比例縮放 + 黑邊補齊，避免拉伸變形。
        letterbox_image = LetterBoxResize(
            self.mask_size,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

        # 🌟 圖片轉換：先縮放到預測區塊大小，再用鏡像 (reflect) 補齊外圍犧牲區
        self.image_transform = transforms.Compose(
            [
                letterbox_image,
                transforms.Pad(pad_size, padding_mode="reflect"),
                transforms.ToTensor(),
            ]
        )
        self.image_unpadded_transform = transforms.Compose(
            [
                letterbox_image,
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = LetterBoxResize(
            self.mask_size,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )

    def __len__(self):
        return len(self.target_names)

    def __getitem__(self, idx):
        file_name = self.target_names[idx]

        if self.use_local:
            image_path = os.path.join(self.local_root, "images", file_name + ".jpg")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Local image not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
        else:
            hf_idx = self.name_to_idx[file_name]
            item = self.ds[hf_idx]
            image = item["image"].convert("RGB")

        # 處理圖片 (輸出為 image_size x image_size)
        orig_w, orig_h = image.size

        if self.split.startswith("test") and not self.return_mask_for_test:
            image_tensor = self.image_transform(image)
            image_unpadded_tensor = self.image_unpadded_transform(image)
            if self.return_unpadded_for_test:
                return (
                    image_tensor,
                    file_name,
                    torch.tensor([orig_h, orig_w]),
                    image_unpadded_tensor,
                )
            return image_tensor, file_name, torch.tensor([orig_h, orig_w])

        # 處理 Mask (輸出為 388x388)
        if self.use_local:
            mask_path = os.path.join(
                self.local_root, "annotations", "trimaps", file_name + ".png"
            )
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Local mask not found: {mask_path}")
            mask = Image.open(mask_path).convert("L")
        else:
            mask = item["mask"].convert("L")

        if self.split == "train":
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                angle = random.uniform(-15.0, 15.0)
                image = TF.rotate(
                    image,
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )
            if random.random() > 0.5:
                # 隨機調整亮度 (0.8~1.2)
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)

                # 隨機調整對比度 (0.8~1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_contrast(image, contrast_factor)

        # augmentation 之後再做 resize/pad，確保 image/mask 幾何同步。
        image_tensor = self.image_transform(image)
        image_unpadded_tensor = self.image_unpadded_transform(image)

        mask = self.mask_transform(mask)
        mask_array = np.array(mask)
        binary_mask = np.zeros_like(mask_array, dtype=np.float32)
        binary_mask[mask_array == 1] = 1.0

        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

        if self.split.startswith("test") and self.return_mask_for_test:
            if self.return_unpadded_for_test:
                return (
                    image_tensor,
                    file_name,
                    mask_tensor,
                    torch.tensor([orig_h, orig_w]),
                    image_unpadded_tensor,
                )
            return image_tensor, file_name, mask_tensor, torch.tensor([orig_h, orig_w])

        # 最終回傳：image_tensor(572x572), mask_tensor(388x388)
        return image_tensor, mask_tensor
