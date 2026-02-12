"""
ComfyUI custom node: OpenPose Rotator.
Rotates OpenPose figures around their torso pivot point.
"""

import torch
import numpy as np

from .pose_utils import rotate_openpose


class OpenPoseRotator:
    CATEGORY = "image/pose"
    FUNCTION = "rotate_pose"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["left", "right"],),
                "degrees": ("INT", {"default": 45, "min": 1, "max": 360}),
            },
            "optional": {
                "pose_keypoint": ("POSE_KEYPOINT",),
            },
        }

    def rotate_pose(
        self,
        image: torch.Tensor,
        direction: str,
        degrees: int,
        pose_keypoint: list | None = None,
    ) -> tuple[torch.Tensor,]:
        """
        Rotate OpenPose figure(s) around torso. Processes batch of images.
        """
        batch_size = image.shape[0]
        results = []

        for i in range(batch_size):
            img = image[i].cpu().numpy()
            # ComfyUI IMAGE: [B,H,W,C] float [0,1]
            h, w = img.shape[:2]

            # Resolve pose_keypoint for this image (may be list per image)
            kp = None
            if pose_keypoint is not None:
                if isinstance(pose_keypoint, list) and len(pose_keypoint) > i:
                    kp = pose_keypoint[i]
                else:
                    kp = pose_keypoint

            out_img, success = rotate_openpose(img, kp, direction, degrees)

            if not success:
                print("OpenPose Rotator: Could not detect torso. Returning input image.")
                out_img = img
            else:
                # pose_utils renders BGR; convert to RGB for ComfyUI
                if out_img.shape[2] == 3:
                    out_img = out_img[..., ::-1]

            # Ensure float [0,1] and 3 channels
            if out_img.dtype != np.float32:
                out_img = out_img.astype(np.float32) / 255.0
            if len(out_img.shape) == 2:
                out_img = np.stack([out_img] * 3, axis=-1)
            results.append(out_img)

        out_tensor = torch.from_numpy(np.stack(results, axis=0)).float()
        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "OpenPoseRotator": OpenPoseRotator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseRotator": "OpenPose Rotator",
}
