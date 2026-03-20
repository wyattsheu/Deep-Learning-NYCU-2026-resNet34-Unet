import torch
import numpy as np


def calculate_dice_score(pred, target):
    """
    TODO: 實作 Dice Score 邏輯
    公式: 2 * (number of common pixels) / (predicted img size + ground truth img size)
    注意: pred 與 target 應該要是 binary masks (0 與 1)
    """
    pred = pred.float()
    target = target.float()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > 0.5).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + 1e-8) / (denominator + 1e-8)
    return dice.mean().item()


# 你可以在這裡加入其他的輔助函式，例如視覺化預測結果的 function


def visualize_predictions(image, pred_mask, target_mask=None, save_path=None):
    """
    視覺化預測結果

    Args:
        image: 輸入圖像 (C, H, W) 或 (H, W)
        pred_mask: 預測的 mask (H, W)
        target_mask: 目標的 mask (H, W)，可選
        save_path: 保存圖像的路徑，如果為 None 則只顯示
    """
    import matplotlib.pyplot as plt

    # 轉換為 numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.cpu().numpy()

    # 處理圖像形狀
    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
        image = np.squeeze(image)

    # 建立圖表
    num_plots = 3 if target_mask is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))

    if num_plots == 2:
        axes = [axes[0], axes[1]]

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    if target_mask is not None:
        axes[2].imshow(target_mask, cmap="gray")
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show(block=True)

    plt.close()


def visualize_predictions_grid(samples):
    """
    將多個預測結果畫在同一張圖上。

    Args:
        samples: list of tuples (image_id, image, pred_mask, target_mask)
    """
    if not samples:
        return

    import matplotlib.pyplot as plt

    num_samples = len(samples)
    has_target = samples[0][3] is not None
    num_cols = 3 if has_target else 2

    # Make a more compact canvas so the whole grid is easier to view on screen.
    cell_w = 3.2
    cell_h = 2.4
    fig, axes = plt.subplots(
        num_samples,
        num_cols,
        figsize=(cell_w * num_cols, cell_h * num_samples),
    )

    # 確保 axes 是二維的，即使只有一個樣本
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (image_id, image, pred_mask, target_mask) in enumerate(samples):
        # 轉換為 numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.cpu().numpy()

        # 處理圖像形狀
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
            image = np.squeeze(image)

        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title(f"Image: {image_id}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_mask, cmap="gray")
        axes[i, 1].set_title("Predicted Mask", fontsize=9)
        axes[i, 1].axis("off")

        if has_target and target_mask is not None:
            axes[i, 2].imshow(target_mask, cmap="gray")
            axes[i, 2].set_title("Ground Truth Mask", fontsize=9)
            axes[i, 2].axis("off")

    plt.tight_layout(pad=0.6, w_pad=0.4, h_pad=0.6)
    plt.show(block=True)
    plt.close()
