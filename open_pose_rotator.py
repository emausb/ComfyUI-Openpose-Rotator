"""
ComfyUI custom node: OpenPose Rotator.
Rotates OpenPose figures around their torso pivot point.
"""

import torch
import numpy as np

from .pose_utils import rotate_openpose, _keypoints_to_openpose_dict


class OpenPoseRotator:
    CATEGORY = "image/pose"
    FUNCTION = "rotate_pose"
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    RETURN_NAMES = ("IMAGE", "POSE_KEYPOINT")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["clockwise", "counterclockwise"],),
                "degrees": ("INT", {"default": 45, "min": 1, "max": 360}),
                "mode": (["simple", "advanced"],),
            },
            "optional": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "debug": ("BOOLEAN", {"default": False}),
                "perspective": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "focal_length": ("FLOAT", {"default": 800.0, "min": 100.0, "max": 5000.0, "step": 50.0}),
            },
        }

    def rotate_pose(
        self,
        image: torch.Tensor,
        direction: str,
        degrees: int,
        mode: str,
        pose_keypoint: list | None = None,
        debug: bool = False,
        perspective: float = 0.0,
        focal_length: float = 800.0,
    ) -> tuple[torch.Tensor, list]:
        """
        Rotate OpenPose figure(s) around torso. Processes batch of images.
        Returns (image_tensor, rotated_pose_keypoints_list).
        """
        batch_size = image.shape[0]
        results = []
        pose_outputs = []

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

            out_img, success, rotated_kp = rotate_openpose(
                img,
                kp,
                direction,
                degrees,
                image_index=i,
                debug=debug,
                mode=mode,
                perspective=perspective,
                focal_length=focal_length,
            )

            if not success:
                print("OpenPose Rotator: Could not detect torso. Returning input image.")
                out_img = img
                pose_outputs.append({"people": []})
            else:
                # pose_utils renders BGR; convert to RGB for ComfyUI
                if out_img.shape[2] == 3:
                    out_img = out_img[..., ::-1]

                # Convert rotated keypoints to OpenPose format for output
                openpose_dict = _keypoints_to_openpose_dict(rotated_kp) if rotated_kp else {}
                pose_outputs.append(openpose_dict)

                # Pre/post keypoints are logged by rotate_openpose in pose_utils

            # Ensure float [0,1] and 3 channels
            if out_img.dtype != np.float32:
                out_img = out_img.astype(np.float32) / 255.0
            if len(out_img.shape) == 2:
                out_img = np.stack([out_img] * 3, axis=-1)
            results.append(out_img)

        out_tensor = torch.from_numpy(np.stack(results, axis=0)).float()
        return (out_tensor, pose_outputs)


NODE_CLASS_MAPPINGS = {
    "OpenPoseRotator": OpenPoseRotator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenPoseRotator": "OpenPose Rotator",
}
