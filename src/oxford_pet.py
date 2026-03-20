import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class OxfordPetDataset(Dataset):
    HF_DATASET_NAME = "wyattsheu/oxford-pet-full-raw"

    # 🌟 新增 image_size 與 mask_size，預設為 UNet 的尺寸
    def __init__(
        self, data_dir="dataset", split="train", image_size=572, mask_size=388
    ):
        self.split = split
        self.data_dir = os.path.abspath(data_dir)
        self.image_size = image_size
        self.mask_size = mask_size

        txt_path = os.path.join(self.data_dir, f"{split}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Split file not found: {txt_path}")

        with open(txt_path, "r") as f:
            self.target_names = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                self.target_names.append(line.split()[0])

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

        # 🌟 圖片轉換：先縮放到預測區塊大小，再用鏡像 (reflect) 補齊外圍犧牲區
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.mask_size, self.mask_size)),
                transforms.Pad(pad_size, padding_mode="reflect"),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.target_names)

    def __getitem__(self, idx):
        file_name = self.target_names[idx]
        hf_idx = self.name_to_idx[file_name]
        item = self.ds[hf_idx]

        # 處理圖片 (輸出為 572x572)
        image = item["image"].convert("RGB")
        image_tensor = self.image_transform(image)

        if self.split.startswith("test"):
            return image_tensor, file_name

        # 處理 Mask (輸出為 388x388)
        mask = item["mask"].convert("L")
        # Mask 只需要 Resize 到目標大小，不需要加上 padding
        mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
        mask_array = np.array(mask)
        binary_mask = np.zeros_like(mask_array, dtype=np.float32)
        binary_mask[mask_array == 1] = 1.0

        mask_tensor = torch.tensor(binary_mask).unsqueeze(0)

        # 最終回傳：image_tensor(572x572), mask_tensor(388x388)
        return image_tensor, mask_tensor
