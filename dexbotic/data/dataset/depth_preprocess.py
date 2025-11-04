from PIL import Image
import torch
import torch.nn.functional as F


class PreprocessDepth:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image) -> torch.Tensor:

        if image is None:
            image = torch.zeros(*self.target_size)
        else:
            image = torch.IntTensor(image)
            image = self.resize_with_padding(image, self.target_size)
        return image

    @staticmethod
    def resize_with_padding(image: torch.IntTensor, target_size):
        """
        Resize and pad a depth image to target size while keeping aspect ratio.

        Args:
            img (Tensor): (H, W), single-channel int tensor
            target_size (tuple): (target_height, target_width)

        Returns:
            Tensor: (target_height, target_width), padded image
        """
        assert img.dim() == 2, "Input must be a 2D tensor (H, W)"

        orig_h, orig_w = img.shape
        target_h, target_w = target_size

        # 计算缩放比例
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))

        # resize（保持数据类型）
        img = img.unsqueeze(0).unsqueeze(0).float()  # shape (1,1,H,W)
        img = F.interpolate(
            img,
            size=(
                new_h,
                new_w),
            mode='bilinear',
            align_corners=False)
        img = img.squeeze(0).squeeze(0).to(torch.int)  # back to int tensor

        # 计算padding大小
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # padding
        img = F.pad(
            img,
            (pad_left,
             pad_right,
             pad_top,
             pad_bottom),
            mode='constant',
            value=0)

        return img
