# OpenPose Rotator

A ComfyUI custom node that rotates OpenPose figures around their torso pivot point. The node accepts an image and optional pose keypoints, detects the torso, applies Y-axis rotation (clockwise/counterclockwise), and outputs the rotated pose image.

## Features

- **Input**: Image (required) + optional POSE_KEYPOINT from OpenPose/DWPose preprocessors
- **Parameters**: Direction (clockwise/counterclockwise), degrees (1-360), mode (simple/advanced)
- **Fallback**: When POSE_KEYPOINT is not provided, uses DWPose from comfyui-controlnet-aux to extract keypoints (requires that extension)
- **Error handling**: Returns input image unchanged if torso cannot be detected
- **Anatomy-aware rotation**: Limb-specific depth scales per OpenPose COCO body indices for more natural turns
- **Adaptive depth**: Depth scale adapts to shoulder width for consistent results across image sizes
- **Face visibility**: Face/head keypoints (nose, eyes, ears) hidden when rotation >90° (back of head)
- **Occlusion**: Points behind the torso are hidden; limbs drawn in depth order for correct overlap

## Installation

1. Clone or copy this folder into your ComfyUI `custom_nodes` directory. Use a valid Python module name (underscores, not hyphens):
   ```
   ComfyUI/custom_nodes/open_pose_rotator/
   ```
   (If your folder is `open-pose-rotator`, rename it to `open_pose_rotator` so Python can import it.)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional, for image-only input) Install [comfyui-controlnet-aux](https://github.com/comfyorg/comfyui-controlnet-aux) for DWPose fallback when pose keypoints are not connected.

## Rotation Modes

- **Simple** (default): Orthographic projection with perspective-modulated depth inference. Best for eye-level views. Use `perspective` to improve results when the source image has a high or low camera angle (e.g. isometric).
- **Advanced**: Perspective projection after rotation; output Y can change; foreshortening applied. Use when you need more realistic 3D behavior for non–eye-level views. Requires tuning `perspective` and `focal_length`.

### Parameters

| Parameter | Simple | Advanced | Description |
|-----------|--------|----------|-------------|
| `perspective` | Yes | Yes | 0 = eye level, positive = camera above (isometric), negative = camera below. Range: -0.5 to 0.5. |
| `focal_length` | Ignored | Yes | Controls perspective strength; higher = flatter, lower = more foreshortening. Default 800, range 100–5000. |

## Usage

1. Add **OpenPose Rotator** from the node menu under **image/pose**
2. Connect an OpenPose/DWPose image to the `image` input
3. Optionally connect POSE_KEYPOINT from DWPose Estimator or OpenPose Pose for better accuracy
4. Set `mode` (simple or advanced)
5. Set `direction` (counterclockwise or clockwise, when viewed from above)
6. Set `degrees` (1-360)
7. For isometric or low-angle source images, adjust `perspective`; for Advanced mode, use `focal_length` to control foreshortening

## Workflow Example

- **With keypoints**: Load Image → DWPose Estimator → OpenPose Rotator (connect both IMAGE and POSE_KEYPOINT) → Apply ControlNet
- **Image only**: Load Image → OpenPose Rotator → Apply ControlNet (requires comfyui-controlnet-aux for DWPose)

## Implementation Notes

Rotation uses geometry-based depth inference from 2D OpenPose keypoints (no ML dependency). Depth is inferred per OpenPose COCO body index using anatomic proportions. **Simple mode** uses orthographic projection with a perspective term that modulates depth by vertical position. **Advanced mode** applies perspective projection after rotation, allowing output Y to change and producing foreshortening. For true 3D pose lifting from 2D, libraries like [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) or [MMPose](https://github.com/open-mmlab/mmpose) exist but use different keypoint formats (e.g. Human3.6M) and would require format conversion.

## License

MIT
