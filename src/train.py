import os
import argparse
import platform

# Linux 上可用 expandable_segments 減少 CUDA 記憶體破碎化；Windows 會噴 warning。
if platform.system() != "Windows":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 只在除錯時手動設為 1，訓練時預設關閉以免速度明顯下降。
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")


from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary


from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate


def dice_loss_from_logits(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits).float()
    targets = targets.float()

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probs.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()


def focal_loss_from_logits(logits, targets, alpha=0.25, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
    )
    # Clamp prevents extreme values from destabilizing exp in mixed precision.
    pt = torch.exp(-torch.clamp(bce_loss, max=100.0))
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def train(
    Epochs=60,
    Batch_size=24,
    Learning_rate=1e-4,
    model_type="ResNet34_UNet",
    show_summary=False,
    max_lr_mult=3.0,
    disable_amp=False,
):
    # 可選擇 "UNet" 或 "ResNet34_UNet"

    project_root = os.path.abspath(os.getcwd())
    data_dir = os.path.join(project_root, "dataset")

    # 🌟 動態設定尺寸，保證兩個模型都能相容
    if model_type == "UNet":
        IMG_SIZE = 572
        MASK_SIZE = 388
    else:
        # 假設 ResNet34_UNet 是有 padding 的卷積，輸入與輸出相同
        IMG_SIZE = 256
        MASK_SIZE = 256

    # 🌟 傳入指定的尺寸給 Dataset
    train_dataset = OxfordPetDataset(
        data_dir=data_dir, split="train", image_size=IMG_SIZE, mask_size=MASK_SIZE
    )
    val_dataset = OxfordPetDataset(
        data_dir=data_dir, split="val", image_size=IMG_SIZE, mask_size=MASK_SIZE
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # Avoid excessive worker processes on limited CPU environments (e.g., Colab).
    dataloader_workers = min(2, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False)

    if model_type == "UNet":
        model = UNet().to(device)
    else:
        model = ResNet34_UNet().to(device)

    if show_summary:
        print("\n" + "=" * 60)
        print(f"🔍 正在掃描 {model_type} 模型內部架構...")
        print("=" * 60)
        # 這裡的 IMG_SIZE 會自動抓取你在上面設定的 572 (UNet) 或 256 (ResNet)
        summary(
            model,
            input_size=(1, 3, IMG_SIZE, IMG_SIZE),
            col_names=["input_size", "output_size", "num_params", "kernel_size"],
            depth=4,
            device=device,
        )
        print("=" * 60 + "\n")

    print(f"device: {device}")
    print(f"training by {model_type} model")
    print(f"batch size: {Batch_size}")

    ###########應註解掉
    # if hasattr(torch, "compile"):
    #     try:
    #         model = torch.compile(model, mode="reduce-overhead")
    #         print("torch.compile enabled")
    #     except Exception as e:
    #         print(f"torch.compile skipped: {e}")

    # # 如果有多個 GPU，使用 nn.DataParallel
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs with nn.DataParallel")
    #     model = nn.DataParallel(model)
    ###########

    # ==========================================
    # 🌟 1. Optimizer 與 梯度累積步數設定
    # ==========================================
    if model_type == "UNet":
        # [UNet 專屬] 改用 AdamW 加 Weight Decay 防過擬合
        optimizer = torch.optim.AdamW(model.parameters(), lr=Learning_rate, weight_decay=1e-2)
        # [UNet 專屬] 累積 4 步 (真實 batch 4 * 累積 4 = 實質 batch 16)
        accumulation_steps = 4 
    else:
        # [ResNet 專屬] 維持原本的設定
        optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
        accumulation_steps = 1 

    # ==========================================
    # 🌟 2. Scheduler 設定
    # ==========================================
    # 計算總步數時，必須考慮到「累積更新」的次數減少了
    total_steps_per_epoch = len(train_loader) // accumulation_steps + (1 if len(train_loader) % accumulation_steps != 0 else 0)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Learning_rate * max_lr_mult,
        steps_per_epoch=total_steps_per_epoch,
        epochs=Epochs,
        # UNet 給 30% 時間暖身，ResNet 預設
        pct_start=0.3 if model_type == "UNet" else 0.3, 
    )

    amp_enabled = use_cuda and (not disable_amp)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    save_dir = os.path.join(project_root, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    best_dice = 0.0

    print(f"max lr (OneCycleLR): {Learning_rate * max_lr_mult:.6g}")
    print(f"amp enabled: {amp_enabled}")
    print(f"gradient accumulation steps: {accumulation_steps}")

    for epoch in range(Epochs):
        # ---------- Training Phase ----------
        model.train()
        train_loss = 0.0
        
        # 確保 epoch 開始前梯度歸零
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Epochs}")
        for batch_idx, (image, mask) in enumerate(pbar):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if amp_enabled and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_context = torch.amp.autocast(device_type="cuda", enabled=True)
            elif amp_enabled:
                autocast_context = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast_context = nullcontext()

            with autocast_context:
                out = model(image)
                if model_type == "UNet":
                    focal_loss = focal_loss_from_logits(out, mask)
                    dice_loss = dice_loss_from_logits(out, mask)
                    loss = 0.5 * focal_loss + 0.5 * dice_loss
                else:
                    loss = nn.BCEWithLogitsLoss()(out, mask)

                # 💡 關鍵 1：Loss 必須除以累積步數，才能算出 4 次的平均梯度
                loss = loss / accumulation_steps

            # 💡 關鍵 2：反向傳播 (累積梯度，但不馬上更新)
            scaler.scale(loss).backward()

            # 💡 關鍵 3：當滿 4 步，或者是最後一個 batch 時，才真正更新權重！
            if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                
                # [UNet 專屬保命符] 梯度裁剪 (防止 Loss 爆炸)
                if model_type == "UNet":
                    scaler.unscale_(optimizer) 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 正式更新權重
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Scheduler 配合真實的更新節奏走
                scheduler.step()

            # 顯示用的 Loss 記得乘回來
            current_loss = loss.item() * accumulation_steps
            train_loss += current_loss
            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        print(f"Evaluating Epoch {epoch+1}...")
        val_dice = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{Epochs}] - Train Loss: {avg_train_loss:.4f} | Val Dice Score: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice

            save_path = os.path.join(save_dir, f"best_{model_type}.pth")
            # # 🔥 神奇的一行：自動適應單卡/多卡環境
            # model_to_save = model.module if hasattr(model, "module") else model
            # torch.save(model_to_save.state_dict(), save_path)
            torch.save(model.state_dict(), save_path)

            print(f"new model saved at {save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model with specified parameters."
    )
    parser.add_argument(
        "--epochs", type=int, default=60, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=24, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["UNet", "ResNet34_UNet"],
        default="ResNet34_UNet",
        help="Type of model to train",
    )
    parser.add_argument(
        "--show_summary",
        action="store_true",
        help="Show torchinfo model summary before training",
    )
    parser.add_argument(
        "--max_lr_mult",
        type=float,
        default=3.0,
        help="OneCycleLR peak multiplier, max_lr = learning_rate * max_lr_mult",
    )
    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable AMP mixed precision for stability debugging",
    )

    args = parser.parse_args()

    train(
        Epochs=args.epochs,
        Batch_size=args.batch_size,
        Learning_rate=args.learning_rate,
        model_type=args.model_type,
        show_summary=args.show_summary,
        max_lr_mult=args.max_lr_mult,
        disable_amp=args.disable_amp,
    )
# python3 src/train.py --epochs 30 --batch_size 24 --learning_rate 0.0001 --model_type UNet or ResNet34_UNet
