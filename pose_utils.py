"""
OpenPose keypoint parsing, torso detection, 3D rotation, and skeleton rendering.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# Body keypoint indices (0-based, ControlNet / DWPose COCO-18 format)
# 0=nose, 1=neck, 2=r_shoulder, 3=r_elbow, 4=r_wrist, 5=l_shoulder, 6=l_elbow, 7=l_wrist,
# 8=r_hip, 9=r_knee, 10=r_ankle, 11=l_hip, 12=l_knee, 13=l_ankle,
# 14=r_eye, 15=l_eye, 16=r_ear, 17=l_ear
# Note: no mid_hip in COCO-18. Legs connect directly from neck to each hip.
TORSO_INDICES = [1, 2, 5, 8, 11]  # neck, r_shoulder, l_shoulder, r_hip, l_hip

# Face/head keypoints (hidden when rotated >150° - back of head visible)
# Keeps face visible when facing to the side (~90°) so at least one eye remains
FACE_INDICES = {0, 14, 15, 16, 17}  # nose, r_eye, l_eye, r_ear, l_ear (COCO-18)
FACE_HIDE_ANGLE = 150  # degrees - hide face only when head is turned away

# Limb-specific depth scales (ControlNet COCO-18 body indices 0-17)
# Torso compact; arms/legs extend further; head moderate
# Based on typical human proportions in frontal view
BODY_DEPTH_SCALES: dict[int, float] = {
    0: 0.30,   # nose
    1: 0.20,   # neck
    2: 0.35,   # r_shoulder
    3: 0.50,   # r_elbow
    4: 0.55,   # r_wrist
    5: 0.35,   # l_shoulder
    6: 0.50,   # l_elbow
    7: 0.55,   # l_wrist
    8: 0.30,   # r_hip (COCO-18)
    9: 0.45,   # r_knee
    10: 0.50,  # r_ankle
    11: 0.30,  # l_hip (COCO-18)
    12: 0.45,  # l_knee
    13: 0.50,  # l_ankle
    14: 0.28,  # r_eye (COCO-18)
    15: 0.28,  # l_eye
    16: 0.30,  # r_ear
    17: 0.30,  # l_ear
}

# Default depth scale for hands (21 keypoints each - no per-index in OpenPose hand spec)
HAND_DEPTH_SCALE = 0.45

# Face mesh (70 keypoints) - same region as head; use head-like depth so face rotates with skull
FACE_DEPTH_SCALE = 0.30

# Static limb definitions: (a, b) connection, draw layer (0=head first, 4=feet last), color index
# Uses ControlNet COCO-18 indices. No mid_hip — legs connect neck→hip directly.
BODY_LIMBS = [
    ((0, 1), 0, 0),   # nose - neck
    ((0, 14), 0, 0),  # nose - r_eye  (COCO-18: 14=r_eye)
    ((0, 15), 0, 0),  # nose - l_eye  (COCO-18: 15=l_eye)
    ((14, 16), 0, 0), # r_eye - r_ear (COCO-18: 16=r_ear)
    ((15, 17), 0, 0), # l_eye - l_ear (COCO-18: 17=l_ear)
    ((1, 2), 0, 0),   # neck - r_shoulder
    ((1, 5), 0, 0),   # neck - l_shoulder
    ((2, 3), 1, 2),   # r_shoulder - r_elbow
    ((3, 4), 2, 3),   # r_elbow - r_wrist
    ((5, 6), 1, 4),   # l_shoulder - l_elbow
    ((6, 7), 2, 5),   # l_elbow - l_wrist
    ((1, 8), 0, 6),   # neck - r_hip  (spine right, COCO-18: 8=r_hip)
    ((1, 11), 0, 6),  # neck - l_hip  (spine left,  COCO-18: 11=l_hip)
    ((8, 9), 3, 7),   # r_hip - r_knee
    ((9, 10), 4, 8),  # r_knee - r_ankle
    ((11, 12), 3, 9), # l_hip - l_knee
    ((12, 13), 4, 10),# l_knee - l_ankle
]

# ControlNet OpenPose standard colors (BGR) - matches expected figure appearance
# Red chest -> orange/yellow arms -> green spine -> teal/blue/purple legs
BODY_COLORS = [
    (0, 0, 255),     # red - neck-shoulders (chest line)
    (0, 0, 255),
    (0, 85, 255),    # orange - r upper arm
    (0, 170, 255),   # yellow - r forearm
    (0, 255, 85),    # green - l upper arm
    (85, 255, 0),    # lime green - l forearm
    (0, 255, 0),     # green - spine (neck to mid_hip)
    (0, 255, 170),   # teal - r thigh
    (0, 255, 255),   # cyan - r knee
    (170, 255, 0),   # light blue - r shin
    (255, 0, 0),     # blue - l thigh
    (255, 85, 0),    # dark blue - l knee
    (255, 0, 170),   # purple - l shin
]

# Static joint definitions: index -> (draw layer, color index) — ControlNet COCO-18
JOINT_LAYERS = {
    0: (0, 0),   # nose
    1: (0, 0),   # neck
    2: (1, 2),   # r_shoulder
    3: (1, 2),   # r_elbow
    4: (1, 3),   # r_wrist
    5: (1, 4),   # l_shoulder
    6: (1, 4),   # l_elbow
    7: (1, 5),   # l_wrist
    8: (2, 7),   # r_hip  (COCO-18)
    9: (2, 8),   # r_knee
    10: (3, 8),  # r_ankle
    11: (2, 9),  # l_hip  (COCO-18)
    12: (2, 10), # l_knee
    13: (3, 10), # l_ankle
    14: (0, 0),  # r_eye  (COCO-18)
    15: (0, 0),  # l_eye
    16: (0, 0),  # r_ear
    17: (0, 0),  # l_ear
}

# Valid keypoint indices for limb drawing (COCO-18: indices 0-17)
MAIN_BODY_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}

# Hand edges (0-indexed, 21 keypoints per hand) - used when hands enabled
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Face mesh edges (0-indexed, 70 keypoints) - OpenPose FACE_PAIRS_RENDER from faceParameters.hpp
FACE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20), (20, 21),
    (22, 23), (23, 24), (24, 25), (25, 26),
    (27, 28), (28, 29), (29, 30),
    (31, 32), (32, 33), (33, 34), (34, 35),
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56),
    (56, 57), (57, 58), (58, 59), (59, 48),
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60),
]


def _keypoints_to_openpose_dict(
    keypoints: dict[str, list[tuple[float, float, float]]],
) -> dict[str, list[float]]:
    """Convert internal keypoint dict to OpenPose/ComfyUI format (flat x,y,c arrays)."""
    body = keypoints.get("body", [])
    hand_left = keypoints.get("hand_left", [])
    hand_right = keypoints.get("hand_right", [])
    face = keypoints.get("face", [])

    def flatten(pts: list[tuple[float, float, float]]) -> list[float]:
        result = []
        for x, y, c in pts:
            result.extend([float(x), float(y), float(c)])
        return result

    person = {
        "pose_keypoints_2d": flatten(body),
        "hand_left_keypoints_2d": flatten(hand_left),
        "hand_right_keypoints_2d": flatten(hand_right),
        "face_keypoints_2d": flatten(face),
    }
    return {"people": [person]}


def _extract_keypoint_array(flat: list[float] | None) -> list[tuple[float, float, float]]:
    """Convert flat [x,y,c, x,y,c, ...] to list of (x, y, conf)."""
    if not flat or len(flat) < 3:
        return []
    points = []
    for i in range(0, len(flat), 3):
        if i + 2 < len(flat):
            x, y, c = flat[i], flat[i + 1], flat[i + 2]
            points.append((float(x), float(y), float(c)))
    return points


def _body_from_numpy(arr: np.ndarray) -> list[tuple[float, float, float]]:
    """Convert body array (N,2) or (N,3) to list of (x,y,conf)."""
    if arr is None or arr.size == 0:
        return []
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return []
    points = []
    for i in range(arr.shape[0]):
        if arr.shape[1] >= 3:
            points.append((float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])))
        elif arr.shape[1] >= 2:
            points.append((float(arr[i, 0]), float(arr[i, 1]), 1.0))
        else:
            break
    return points


def _scale_keypoints_to_image(
    keypoints: dict[str, list[tuple[float, float, float]]],
    width: int,
    height: int,
) -> dict[str, list[tuple[float, float, float]]]:
    """
    Scale keypoints to image dimensions.
    - Normalized [0,1] coords: scale by width, height
    - Pixel coords from smaller resolution (e.g. DWPose at 384): scale to match image
    """
    all_x, all_y = [], []
    for pts in keypoints.values():
        for x, y, c in pts:
            if c > 0:
                all_x.append(x)
                all_y.append(y)

    if not all_x or not all_y:
        return keypoints

    max_x, max_y = max(all_x), max(all_y)
    min_x, min_y = min(all_x), min(all_y)

    # Normalized coords in [0,1] (ComfyUI POSE_KEYPOINT may use this)
    if max_x <= 1.5 and max_y <= 1.5 and min_x >= -0.1 and min_y >= -0.1:
        scale_x, scale_y = width, height
    # Pixel coords from smaller resolution (e.g. DWPose at 384)
    # Only scale when max is clearly from a smaller resolution; avoid scaling when
    # the figure is already at image size but doesn't fill the frame (causes ~40% blow-up)
    elif max_x < width * 0.76 and max_y < height * 0.76:
        scale_x = width / max(max_x, 1)
        scale_y = height / max(max_y, 1)
        if abs(scale_x - 1.0) < 0.01 and abs(scale_y - 1.0) < 0.01:
            return keypoints
    else:
        return keypoints

    result: dict[str, list[tuple[float, float, float]]] = {}
    for part, pts in keypoints.items():
        scaled = [
            (x * scale_x, y * scale_y, c) if c > 0 else (x, y, c)
            for x, y, c in pts
        ]
        result[part] = scaled
    return result


def parse_pose_keypoint(pose_keypoint: Any) -> dict[str, list[tuple[float, float, float]]] | None:
    """
    Parse POSE_KEYPOINT from ComfyUI (OpenPose/DWPose format) into normalized structure.
    Returns dict with keys: body, hand_left, hand_right, face; each value is list of (x,y,conf).
    Returns None if invalid or empty.
    """
    if pose_keypoint is None:
        return None

    # POSE_KEYPOINT can be a list (one per image) or single dict
    data = pose_keypoint
    if isinstance(pose_keypoint, list):
        if not pose_keypoint:
            return None
        data = pose_keypoint[0]

    if not isinstance(data, dict):
        return None

    # Structure 1: {"people": [{"pose_keypoints_2d": [...], "hand_left_keypoints_2d": [...], ...}]}
    people = data.get("people", [])
    if people:
        person = people[0]
        body = _extract_keypoint_array(person.get("pose_keypoints_2d"))
        hand_left = _extract_keypoint_array(person.get("hand_left_keypoints_2d"))
        hand_right = _extract_keypoint_array(person.get("hand_right_keypoints_2d"))
        face = _extract_keypoint_array(person.get("face_keypoints_2d"))
        if body:
            return {"body": body, "hand_left": hand_left, "hand_right": hand_right, "face": face}

    # Structure 2: {"person_0": {"body": np.array, "hands": np.array, "face": np.array}}
    for key in ("person_0", "person"):
        if key in data:
            person = data[key]
            body = person.get("body")
            if body is not None:
                body = _body_from_numpy(np.asarray(body))
            else:
                body = []
            hands = person.get("hands", np.zeros((2, 21, 2)))
            hands = np.asarray(hands)
            hand_left = _body_from_numpy(hands[0]) if hands.shape[0] > 0 else []
            hand_right = _body_from_numpy(hands[1]) if hands.shape[0] > 1 else []
            face = _body_from_numpy(np.asarray(person.get("face", [])))
            if body:
                return {"body": body, "hand_left": hand_left, "hand_right": hand_right, "face": face}

    # Structure 3: flat {"pose_keypoints_2d": [...], ...} at top level (no "people")
    body = _extract_keypoint_array(data.get("pose_keypoints_2d"))
    if body:
        hand_left = _extract_keypoint_array(data.get("hand_left_keypoints_2d"))
        hand_right = _extract_keypoint_array(data.get("hand_right_keypoints_2d"))
        face = _extract_keypoint_array(data.get("face_keypoints_2d"))
        return {"body": body, "hand_left": hand_left, "hand_right": hand_right, "face": face}

    return None


def extract_keypoints_from_image(image: np.ndarray) -> dict[str, list[tuple[float, float, float]]] | None:
    """
    Run DWPose on image to extract keypoints. Requires comfyui-controlnet-aux.
    Returns parsed keypoint dict or None on failure.
    """
    try:
        from custom_controlnet_aux.dwpose import DwposeDetector
        import comfy.model_management as model_management
    except ImportError:
        return None

    try:
        model = DwposeDetector.from_pretrained(
            "yzd-v/DWPose",
            "yzd-v/DWPose",
            det_filename="yolox_l.onnx",
            pose_filename="dw-ll_ucoco_384.onnx",
            torchscript_device=model_management.get_torch_device(),
        )
        # DWPose expects HWC uint8; annotators often expect BGR
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pose_img, openpose_dict = model(
            image,
            include_hand=True,
            include_face=True,
            include_body=True,
            detect_resolution=min(image.shape[0], image.shape[1], 512),
        )
        del model
        return parse_pose_keypoint(openpose_dict)
    except Exception:
        return None


def compute_torso_pivot(keypoints: dict[str, list[tuple[float, float, float]]]) -> tuple[float, float] | None:
    """
    Compute torso center from body keypoints. Uses neck, shoulders, mid_hip, hips.
    Supports "no torso segment" variants: derives neck from shoulders and mid_hip from hips
    when those keypoints are missing (c<=0) but shoulders/hips are present.
    Returns (cx, cy) or None if insufficient keypoints.
    """
    body = keypoints.get("body", [])
    if len(body) < 2:
        return None

    def get_pt(idx: int) -> tuple[float, float, float] | None:
        if idx >= len(body):
            return None
        x, y, c = body[idx]
        return (x, y, c) if c > 0 else None

    xs, ys, total_conf = 0.0, 0.0, 0.0

    for idx in TORSO_INDICES:
        pt = get_pt(idx)
        if pt is not None:
            x, y, c = pt
            xs += x * c
            ys += y * c
            total_conf += c

    # Fallback: derive neck from shoulders if missing
    # COCO-18 has no mid_hip; both hips (8=r_hip, 11=l_hip) are already in TORSO_INDICES
    if get_pt(1) is None:
        r_sh = get_pt(2)
        l_sh = get_pt(5)
        if r_sh is not None and l_sh is not None:
            nx = (r_sh[0] + l_sh[0]) / 2
            ny = (r_sh[1] + l_sh[1]) / 2
            xs += nx
            ys += ny
            total_conf += 1.0

    if total_conf <= 0:
        return None

    return (xs / total_conf, ys / total_conf)


def _compute_adaptive_depth_scale(
    keypoints: dict[str, list[tuple[float, float, float]]],
    pivot: tuple[float, float],
) -> float:
    """
    Compute adaptive depth scale from body proportions (shoulder width).
    Normalizes to typical pose so rotation looks consistent across image sizes.
    """
    body = keypoints.get("body", [])
    if len(body) < 6:
        return 0.4
    x2, _, c2 = body[2]  # r_shoulder
    x5, _, c5 = body[5]   # l_shoulder
    if c2 <= 0 or c5 <= 0:
        return 0.4
    shoulder_width = abs(x2 - x5)
    if shoulder_width < 1:
        return 0.4
    # Typical shoulder width ~100-150px in normalized poses; scale so base 0.4 holds
    return 0.4 * (120.0 / shoulder_width)


# Perspective scale_y: bridges the raw (cy - y) pixel value to z-units.
# In advanced mode the coefficient is derived geometrically from ADVANCED_CAMERA_ELEVATION_DEG;
# this constant keeps the formula consistent: z += perspective * (cy - y) * PERSPECTIVE_SCALE_Y.
PERSPECTIVE_SCALE_Y = 0.3

# Advanced mode: fixed physical camera / figure assumptions.
# Camera is elevated 35° above horizontal, looking at a standing figure that fills
# roughly 80-100% (average ~90%) of the image height.
ADVANCED_CAMERA_ELEVATION_DEG = 35.0
ADVANCED_FIGURE_FILL = 0.90

# Ankle y-parity threshold: fraction of figure height (neck→lower ankle) within which both ankles
# are considered to be at the same elevation (standing upright on flat ground).
UPRIGHT_ANKLE_THRESHOLD = 0.08


def _detect_upright_figure(
    keypoints: dict[str, list[tuple[float, float, float]]],
) -> tuple[bool, tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """
    Return (is_upright, neck_pt, ankle_mid_pt).
    Upright is True when neck (1) and both ankles (11=r, 14=l) are detected and both ankle y values
    differ by less than UPRIGHT_ANKLE_THRESHOLD * figure_height (neck y to lower ankle y).
    """
    body = keypoints.get("body", [])

    def get_pt(idx: int) -> tuple[float, float, float] | None:
        if idx >= len(body):
            return None
        x, y, c = body[idx]
        return (x, y, c) if c > 0 else None

    neck = get_pt(1)
    r_ankle = get_pt(10)  # COCO-18: r_ankle=10
    l_ankle = get_pt(13)  # COCO-18: l_ankle=13

    if neck is None or r_ankle is None or l_ankle is None:
        return False, None, None

    lower_ankle_y = max(r_ankle[1], l_ankle[1])
    figure_height = abs(lower_ankle_y - neck[1])
    if figure_height < 1:
        return False, None, None

    ankle_y_diff = abs(r_ankle[1] - l_ankle[1])
    if ankle_y_diff > UPRIGHT_ANKLE_THRESHOLD * figure_height:
        return False, None, None

    ankle_mid: tuple[float, float, float] = (
        (r_ankle[0] + l_ankle[0]) / 2,
        (r_ankle[1] + l_ankle[1]) / 2,
        1.0,
    )
    return True, neck, ankle_mid


def _compute_advanced_params(image_height: int) -> tuple[float, float]:
    """
    Derive perspective and focal_length for advanced mode from fixed camera/figure assumptions.

    Camera elevation θ = ADVANCED_CAMERA_ELEVATION_DEG (35°) above horizontal.
    Figure fills ADVANCED_FIGURE_FILL (~90%) of image height.

    perspective: maps vertical pixel distance from pivot to depth.
        Formula: z += perspective * (cy - y) * PERSPECTIVE_SCALE_Y
        For geometric accuracy: perspective * PERSPECTIVE_SCALE_Y = tan(θ)
        → perspective = tan(35°) / 0.3 ≈ 2.33

    focal_length: perspective projection strength, scaled to image resolution.
        From pinhole geometry: focal_length = image_height * fill / tan(θ)
        At 35° with 90% fill → focal_length ≈ 1.29 × image_height
    """
    tan_elev = math.tan(math.radians(ADVANCED_CAMERA_ELEVATION_DEG))
    perspective = tan_elev / PERSPECTIVE_SCALE_Y
    focal_length = image_height * ADVANCED_FIGURE_FILL / tan_elev
    return perspective, focal_length


def rotate_keypoints_3d(
    keypoints: dict[str, list[tuple[float, float, float]]],
    pivot: tuple[float, float],
    degrees: float,
    direction: str,
    depth_scale: float | None = None,
    mode: str = "simple",
    perspective: float = 0.0,
    focal_length: float = 800.0,
) -> dict[str, list[tuple[float, float, float]]]:
    """
    Rotate all keypoints around Y-axis through pivot. Counterclockwise = positive angle, clockwise = negative.
    Uses limb-specific depth inference per OpenPose COCO body indices for 3D-style projection.

    mode: "simple" = orthographic output with perspective-modulated depth; "advanced" = perspective projection.
    perspective: 0 = eye level, positive = camera above (isometric), negative = camera below.
    focal_length: (advanced only) controls perspective strength; higher = flatter, lower = more foreshortening.

    In advanced mode, when the figure is detected as standing upright (neck + matching-y ankles),
    the rotation axis is the 3D spine computed from the ankle midpoint through the neck rather than
    a pure vertical Y-axis. This keeps the neck stationary and tilts the axis to match the figure's lean.
    """
    theta_deg = degrees if direction == "counterclockwise" else -degrees
    theta = math.radians(theta_deg)
    cx, cy = pivot

    if depth_scale is None:
        depth_scale = _compute_adaptive_depth_scale(keypoints, pivot)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    use_perspective_projection = mode == "advanced"

    # --- Advanced mode: detect upright figure and build spine axis ---
    # axis_point and axis_dir are in pivot-relative 3D space (origin = pivot).
    axis_point: np.ndarray | None = None
    axis_dir: np.ndarray | None = None

    # Upright spine axis applies in both modes — neck is always the rotation anchor when conditions met
    is_upright, neck_pt, ankle_mid_pt = _detect_upright_figure(keypoints)
    if is_upright and neck_pt is not None and ankle_mid_pt is not None:
            def _infer_z(px: float, py: float, limb_scale: float) -> float:
                return (px - cx) * depth_scale * limb_scale + perspective * (cy - py) * PERSPECTIVE_SCALE_Y

            # Neck 3D (pivot-relative)
            nz = _infer_z(neck_pt[0], neck_pt[1], BODY_DEPTH_SCALES.get(1, 0.20))
            neck_3d = np.array([neck_pt[0] - cx, neck_pt[1] - cy, nz])

            # Ankle midpoint 3D (pivot-relative) — average depth of both ankles (COCO-18: 10/13)
            r_scale = BODY_DEPTH_SCALES.get(10, 0.50)
            l_scale = BODY_DEPTH_SCALES.get(13, 0.50)
            body = keypoints.get("body", [])
            r_ankle = body[10]
            l_ankle = body[13]
            z_r = _infer_z(r_ankle[0], r_ankle[1], r_scale)
            z_l = _infer_z(l_ankle[0], l_ankle[1], l_scale)
            ankle_3d = np.array([
                ankle_mid_pt[0] - cx,
                ankle_mid_pt[1] - cy,
                (z_r + z_l) / 2,
            ])

            spine = neck_3d - ankle_3d
            spine_norm = float(np.linalg.norm(spine))
            if spine_norm > 1e-6:
                axis_dir = spine / spine_norm
                axis_point = neck_3d  # axis passes through neck

    def _infer_z_point(x: float, y: float, scale: float) -> float:
        # scale is depth_scale * limb_scale
        return (x - cx) * scale + perspective * (cy - y) * PERSPECTIVE_SCALE_Y

    def rotate_point(
        x: float, y: float, scale: float
    ) -> tuple[float, float, float]:
        z = _infer_z_point(x, y, scale)

        if axis_dir is not None and axis_point is not None:
            # Rodrigues rotation around the spine axis (advanced + upright)
            p = np.array([x - cx, y - cy, z])
            p_rel = p - axis_point
            k = axis_dir
            p_rot = (p_rel * cos_t
                     + np.cross(k, p_rel) * sin_t
                     + k * float(np.dot(k, p_rel)) * (1.0 - cos_t))
            p_world = p_rot + axis_point
            x_rot, y_rot_3d, z_rot = float(p_world[0]), float(p_world[1]), float(p_world[2])
            if use_perspective_projection:
                z_cam = focal_length + z_rot
                z_cam = max(z_cam, 0.1 * focal_length)
                # Perspective division applies only to X (horizontal foreshortening during yaw).
                # Y is kept orthographic: dividing by z_cam would vertically stretch the figure
                # because the elevated camera assigns large z values to high/low keypoints,
                # causing them to shift far outside the image even at near-zero rotation angles.
                return (cx + focal_length * x_rot / z_cam,
                        cy + y_rot_3d,
                        z_rot)
            return (x_rot + cx, y_rot_3d + cy, z_rot)

        # Default: Y-axis rotation in XZ plane (simple mode or non-upright advanced)
        x_rot = (x - cx) * cos_t - z * sin_t
        z_rot = (x - cx) * sin_t + z * cos_t
        if use_perspective_projection:
            z_cam = focal_length + z_rot
            z_cam = max(z_cam, 0.1 * focal_length)
            return (cx + focal_length * x_rot / z_cam,
                    y,  # Y unchanged: pure yaw does not affect vertical screen position
                    z_rot)
        return (x_rot + cx, y, z_rot)

    result: dict[str, list[tuple[float, float, float]]] = {}

    for part, points in keypoints.items():
        rotated: list[tuple[float, float, float]] = []

        for i, (x, y, c) in enumerate(points):
            if c <= 0:
                rotated.append((x, y, c))
                continue

            if part == "body":
                scale = BODY_DEPTH_SCALES.get(i, HAND_DEPTH_SCALE)
            elif part == "face":
                scale = FACE_DEPTH_SCALE
            else:
                scale = HAND_DEPTH_SCALE
            nx, ny, _ = rotate_point(x, y, depth_scale * scale)
            rotated.append((nx, ny, c))

        result[part] = rotated

    # Face visibility — two regimes:
    #
    # Forward-facing (both eyes detected in original pose):
    #   Hemisphere model — a face keypoint whose original normalised lateral position is
    #   t = (x − face_cx) / half_w is hidden once  sign_dir × t > cos(|θ|).
    #   This is geometrically exact for a hemispherical head: edge features (ears) disappear
    #   first; the nose disappears last (only past 90°).
    #   sign_dir = +1 for CCW (right-of-image side recedes), −1 for CW (left side recedes).
    #
    # Non-forward-facing (one or both eyes absent):
    #   Original binary threshold — hide all face points once |θ| ≥ FACE_HIDE_ANGLE.

    if abs(theta_deg) > 0:
        orig_body = keypoints.get("body", [])
        orig_face  = keypoints.get("face", [])

        r_eye_orig = orig_body[14] if len(orig_body) > 14 else None
        l_eye_orig = orig_body[15] if len(orig_body) > 15 else None
        both_eyes  = (
            r_eye_orig is not None and r_eye_orig[2] > 0
            and l_eye_orig is not None and l_eye_orig[2] > 0
        )

        if both_eyes:
            # Face centre and half-width from original pose.
            # Prefer the face mesh (covers full jaw); fall back to eye spacing.
            face_xs = [x for x, y, c in orig_face if c > 0] if orig_face else []
            if len(face_xs) >= 2:
                face_cx_orig = (min(face_xs) + max(face_xs)) / 2.0
                half_w = (max(face_xs) - min(face_xs)) / 2.0
            else:
                # Eyes span roughly the middle third of the face, so half_w ≈ 1.5 × eye_sep
                face_cx_orig = (r_eye_orig[0] + l_eye_orig[0]) / 2.0
                half_w = abs(l_eye_orig[0] - r_eye_orig[0]) * 1.5
            half_w = max(half_w, 1.0)

            cos_theta = math.cos(math.radians(abs(theta_deg)))
            sign_dir  = 1.0 if theta_deg > 0 else -1.0  # +1 CCW, −1 CW

            # Hide face mesh keypoints (70-point OpenPose face)
            if "face" in result and orig_face:
                face_out = result["face"]
                for i, (x_orig, y_orig, c_orig) in enumerate(orig_face):
                    if c_orig <= 0 or i >= len(face_out):
                        continue
                    t = (x_orig - face_cx_orig) / half_w
                    if sign_dir * t > cos_theta:
                        px, py, _ = face_out[i]
                        face_out[i] = (px, py, 0.0)

            # Hide body face keypoints: nose(0), r_eye(14), l_eye(15), r_ear(16), l_ear(17)
            if "body" in result:
                body_out = result["body"]
                for idx in FACE_INDICES:
                    if idx >= len(orig_body) or idx >= len(body_out):
                        continue
                    ox, oy, oc = orig_body[idx]
                    if oc <= 0:
                        continue
                    t = (ox - face_cx_orig) / half_w
                    if sign_dir * t > cos_theta:
                        px, py, _ = body_out[idx]
                        body_out[idx] = (px, py, 0.0)

        else:
            # Non-forward-facing: original binary threshold
            if abs(theta_deg) >= FACE_HIDE_ANGLE:
                if "body" in result:
                    body = result["body"]
                    for idx in FACE_INDICES:
                        if idx < len(body):
                            x, y, c = body[idx]
                            body[idx] = (x, y, 0.0)
                if "face" in result:
                    result["face"] = [(x, y, 0.0) for x, y, c in result["face"]]

    return result


def _get_anchor_point(keypoints: dict[str, list[tuple[float, float, float]]]) -> tuple[float, float] | None:
    """
    Get anchor point from body keypoints for positioning.
    Prefer neck (1); if missing, use midpoint of shoulders (2, 5); if only one shoulder, use it.
    Returns (x, y) or None if insufficient keypoints.
    """
    body = keypoints.get("body", [])
    if len(body) < 2:
        return None

    def get_pt(idx: int) -> tuple[float, float] | None:
        if idx >= len(body):
            return None
        x, y, c = body[idx]
        return (x, y) if c > 0 else None

    # 1. Neck (index 1)
    neck = get_pt(1)
    if neck is not None:
        return neck

    # 2. Both shoulders (2, 5)
    r_sh = get_pt(2)
    l_sh = get_pt(5)
    if r_sh is not None and l_sh is not None:
        return ((r_sh[0] + l_sh[0]) / 2, (r_sh[1] + l_sh[1]) / 2)

    # 3. Single shoulder
    if r_sh is not None:
        return r_sh
    if l_sh is not None:
        return l_sh

    return None


def _position_keypoints_by_anchor(
    rotated: dict[str, list[tuple[float, float, float]]],
    original: dict[str, list[tuple[float, float, float]]],
) -> dict[str, list[tuple[float, float, float]]]:
    """
    Translate rotated keypoints so the anchor (neck or shoulders) matches the original position.
    Keeps the figure anchored in place based on its pre-rotation position.
    """
    orig_anchor = _get_anchor_point(original)
    rot_anchor = _get_anchor_point(rotated)
    if orig_anchor is None or rot_anchor is None:
        return rotated

    dx = orig_anchor[0] - rot_anchor[0]
    dy = orig_anchor[1] - rot_anchor[1]

    result: dict[str, list[tuple[float, float, float]]] = {}
    for part, pts in rotated.items():
        result[part] = [(x + dx, y + dy, c) for x, y, c in pts]
    return result


def join_broken_segments(
    keypoints: dict[str, list[tuple[float, float, float]]],
    threshold: float,
) -> dict[str, list[tuple[float, float, float]]]:
    """
    Optional: join broken segments when one endpoint visible, other missing.
    Currently a no-op (disabled) per plan.
    """
    return keypoints


def _draw_dashed_line(
    canvas: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 8,
    gap_len: int = 6,
) -> None:
    """Draw a dashed line through p0 and p1, extended to canvas edges in both directions."""
    h, w = canvas.shape[:2]
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length

    def t_to_edge(ox: float, oy: float, vx: float, vy: float) -> float:
        t = float("inf")
        if vx > 1e-9:
            t = min(t, (w - 1 - ox) / vx)
        elif vx < -1e-9:
            t = min(t, -ox / vx)
        if vy > 1e-9:
            t = min(t, (h - 1 - oy) / vy)
        elif vy < -1e-9:
            t = min(t, -oy / vy)
        return max(t, 0.0)

    t_fwd = t_to_edge(x0, y0, ux, uy)
    t_bwd = t_to_edge(x0, y0, -ux, -uy)
    sx = x0 - t_bwd * ux
    sy = y0 - t_bwd * uy
    total = t_fwd + t_bwd

    t = 0.0
    drawing = True
    while t < total:
        seg = dash_len if drawing else gap_len
        t_end = min(t + seg, total)
        if drawing:
            fx, fy = sx + t * ux, sy + t * uy
            tx, ty = sx + t_end * ux, sy + t_end * uy
            cv2.line(canvas, (int(fx), int(fy)), (int(tx), int(ty)), color, thickness)
        t = t_end
        drawing = not drawing


def render_openpose_image(
    keypoints: dict[str, list[tuple[float, float, float]]],
    width: int,
    height: int,
    debug: bool = False,
    axis_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Draw OpenPose skeleton on blank canvas. Black background (matches ControlNet pose images).
    Uses static connection and draw-order definitions; no depth computation.
    When debug=True and axis_line is provided, draws the rotation axis as a white dashed line
    through the two given 2D points (extended to canvas edges).
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    body = keypoints.get("body", [])

    # Build limb list from static definitions; filter by validity
    max_limb_ratio = 0.5  # Max limb length vs image diagonal; filters bad connections
    diag = (width**2 + height**2) ** 0.5
    max_limb_len = diag * max_limb_ratio

    limb_data: list[tuple[int, int, int, int, float]] = []
    for i, ((a, b), layer, color_idx) in enumerate(BODY_LIMBS):
        if a >= len(body) or b >= len(body):
            continue
        if a not in MAIN_BODY_INDICES or b not in MAIN_BODY_INDICES:
            continue
        x1, y1, c1 = body[a]
        x2, y2, c2 = body[b]
        if c1 <= 0 or c2 <= 0:
            continue
        limb_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if limb_len > max_limb_len:
            continue
        # Use max_x: limbs extending right (front when rotated) drawn last
        max_x = max(x1, x2)
        limb_data.append((a, b, layer, color_idx, max_x))

    # Draw limbs: neutral gray (color is on nodes for debugging)
    limb_data.sort(key=lambda t: (t[2], t[4]))
    limb_color = (128, 128, 128)  # gray
    for a, b, _, _, _ in limb_data:
        x1, y1, c1 = body[a]
        x2, y2, c2 = body[b]
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), limb_color, 2)

    # Draw joints: colored by index, twice as large for debugging
    joint_radius = 8
    joint_data = [
        (i, x, y, c)
        for i, (x, y, c) in enumerate(body)
        if c > 0 and i in JOINT_LAYERS
    ]
    joint_data.sort(key=lambda t: (JOINT_LAYERS.get(t[0], (0, 0))[0], t[1]))
    for i, x, y, c in joint_data:
        _, color_idx = JOINT_LAYERS.get(i, (0, 0))
        color = BODY_COLORS[color_idx]
        cv2.circle(canvas, (int(x), int(y)), joint_radius, color, -1)

    # Debug: draw rotation axis (white, dashed, thin)
    if debug and axis_line is not None:
        _draw_dashed_line(canvas, axis_line[0], axis_line[1], (255, 255, 255), thickness=1)

    # Face mesh (OpenPose 70-point) - draw before hands so body/face compose first
    face_pts = keypoints.get("face", [])
    if len(face_pts) >= 2:
        face_color = (180, 180, 180)  # light gray
        face_max_len = diag * 0.4
        for a, b in FACE_EDGES:
            if a >= len(face_pts) or b >= len(face_pts):
                continue
            x1, y1, c1 = face_pts[a]
            x2, y2, c2 = face_pts[b]
            if c1 <= 0 or c2 <= 0:
                continue
            limb_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if limb_len > face_max_len:
                continue
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), face_color, 1)
        # Draw face keypoints (small circles)
        for x, y, c in face_pts:
            if c > 0:
                cv2.circle(canvas, (int(x), int(y)), 2, face_color, -1)

    # Hands (21 keypoints each)
    hand_color = (150, 150, 150)
    hand_max_len = diag * 0.3
    for hand_pts in [keypoints.get("hand_left", []), keypoints.get("hand_right", [])]:
        if len(hand_pts) < 2:
            continue
        for a, b in HAND_EDGES:
            if a >= len(hand_pts) or b >= len(hand_pts):
                continue
            x1, y1, c1 = hand_pts[a]
            x2, y2, c2 = hand_pts[b]
            if c1 <= 0 or c2 <= 0:
                continue
            limb_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if limb_len > hand_max_len:
                continue
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), hand_color, 1)
        for x, y, c in hand_pts:
            if c > 0:
                cv2.circle(canvas, (int(x), int(y)), 2, hand_color, -1)

    return canvas


def rotate_openpose(
    image: np.ndarray,
    pose_keypoint: Any | None,
    direction: str,
    degrees: float,
    image_index: int = 0,
    debug: bool = False,
    mode: str = "simple",
    recenter: bool = False,
) -> tuple[np.ndarray, bool, dict[str, list[tuple[float, float, float]]] | None]:
    """
    Main pipeline: extract/parse keypoints, detect torso, rotate, render.
    Returns (output_image, success, rotated_keypoints). On failure, returns (input_image, False, None).
    rotated_keypoints is internal format: {"body": [(x,y,c),...], "hand_left": [...], ...}
    image_index: used for console logging when processing batches.

    mode: "simple" uses orthographic projection at eye level (perspective=0).
          "advanced" uses perspective projection with camera parameters derived from
          ADVANCED_CAMERA_ELEVATION_DEG (35°) and ADVANCED_FIGURE_FILL (90%), which
          gives geometrically correct depth inference and focal length for the assumed pose.
    recenter: when True, horizontally centers the rotated figure within the image after all
              other positioning corrections. Centering is measured from the widest body
              keypoints; vertical position is not changed.
    """
    h, w = image.shape[:2]

    # Derive camera parameters from mode and fixed assumptions
    if mode == "advanced":
        perspective, focal_length = _compute_advanced_params(h)
    else:
        perspective = 0.0
        focal_length = 800.0  # unused in simple (orthographic) mode

    # Get keypoints
    if pose_keypoint is not None:
        keypoints = parse_pose_keypoint(pose_keypoint)
    else:
        keypoints = extract_keypoints_from_image(image)

    if keypoints is None:
        return image, False, None

    # Scale normalized coords to image size (ComfyUI may pass 0-1 range)
    keypoints = _scale_keypoints_to_image(keypoints, w, h)

    # --- Stance detection (always logged) ---
    # Compares the y-positions of both ankles relative to figure height.
    # Within UPRIGHT_ANKLE_THRESHOLD → both feet on ground (spin axis = spine).
    # Outside threshold        → one foot on ground (spin axis = that ankle).
    _body_kp = keypoints.get("body", [])

    def _stance_pt(idx: int) -> tuple[float, float, float] | None:
        if idx >= len(_body_kp):
            return None
        x, y, c = _body_kp[idx]
        return (x, y, c) if c > 0 else None

    _neck_s = _stance_pt(1)
    _r_ank_s = _stance_pt(10)  # COCO-18: r_ankle=10
    _l_ank_s = _stance_pt(13)  # COCO-18: l_ankle=13

    stance: str = "unknown"
    ground_ankle_pt: tuple[float, float, float] | None = None
    ground_ankle_idx: int | None = None

    if _neck_s is not None and _r_ank_s is not None and _l_ank_s is not None:
        _lower_y = max(_r_ank_s[1], _l_ank_s[1])
        _fig_h = abs(_lower_y - _neck_s[1])
        if _fig_h >= 1:
            _ank_diff = abs(_r_ank_s[1] - _l_ank_s[1])
            if _ank_diff <= UPRIGHT_ANKLE_THRESHOLD * _fig_h:
                stance = "both_feet"
                print(f"[OpenPose Rotator] Image {image_index}: both feet on ground.")
            else:
                stance = "one_foot"
                # Ground foot = the lower ankle (larger y, since y increases downward)
                if _r_ank_s[1] >= _l_ank_s[1]:
                    ground_ankle_pt = _r_ank_s
                    ground_ankle_idx = 10
                else:
                    ground_ankle_pt = _l_ank_s
                    ground_ankle_idx = 13
                print(
                    f"[OpenPose Rotator] Image {image_index}: one foot on ground "
                    f"(ankle idx {ground_ankle_idx} at "
                    f"{ground_ankle_pt[0]:.0f}, {ground_ankle_pt[1]:.0f})."
                )

    # Console: pre-rotated figure (before rotation, after scaling) - only when debug
    if debug:
        pre_dict = _keypoints_to_openpose_dict(keypoints)
        face_pts = keypoints.get("face", [])
        face_visible = sum(1 for _, _, c in face_pts if c > 0)
        if face_visible > 0:
            print(f"[OpenPose Rotator] Image {image_index}: face mesh present ({face_visible}/{len(face_pts)} visible keypoints).")
        else:
            print(f"[OpenPose Rotator] Image {image_index}: no face mesh in input.")
        print(f"[OpenPose Rotator] Image {image_index} PRE-ROTATED (direction={direction}, degrees={degrees}):")
        print(f"  body: {keypoints.get('body', [])}")
        print(f"  pose_keypoints_2d (flat): {pre_dict.get('people', [{}])[0].get('pose_keypoints_2d', [])}")

    # Torso pivot — for one-foot stance, shift the rotation axis to the ground ankle's x
    torso_pivot = compute_torso_pivot(keypoints)
    if torso_pivot is None:
        return image, False, None

    if stance == "one_foot" and ground_ankle_pt is not None:
        # cx = ground ankle; cy stays at torso level for depth-inference in z-calculation
        pivot = (ground_ankle_pt[0], torso_pivot[1])
    else:
        pivot = torso_pivot

    # Rotate keypoints around the chosen pivot
    rotated = rotate_keypoints_3d(
        keypoints,
        pivot,
        degrees,
        direction,
        mode=mode,
        perspective=perspective,
        focal_length=focal_length,
    )
    rotated = join_broken_segments(rotated, 0.0)

    # Position by anchor
    # • both_feet / unknown  → align neck/shoulders (original behaviour)
    # • one_foot             → lock the ground ankle so it stays pinned to its original position
    orig_anchor = _get_anchor_point(keypoints)
    rot_anchor = _get_anchor_point(rotated)

    if stance == "one_foot" and ground_ankle_pt is not None and ground_ankle_idx is not None:
        body_rot = rotated.get("body", [])
        if ground_ankle_idx < len(body_rot) and body_rot[ground_ankle_idx][2] > 0:
            rot_ax, rot_ay, _ = body_rot[ground_ankle_idx]
            dx = ground_ankle_pt[0] - rot_ax
            dy = ground_ankle_pt[1] - rot_ay
            rotated = {
                part: [(px + dx, py + dy, c) for px, py, c in pts]
                for part, pts in rotated.items()
            }
        else:
            rotated = _position_keypoints_by_anchor(rotated, keypoints)
    else:
        rotated = _position_keypoints_by_anchor(rotated, keypoints)

    # Horizontal in-frame correction for one-foot stance.
    # After pinning the ankle, the swept side of the figure can overflow the image edge.
    # When that happens, shift the *entire* figure (ankle included) just far enough to
    # bring the overflowing side back to the image boundary.
    # Only body keypoints are used to measure overflow — hand / face extremities are ignored.
    if stance == "one_foot":
        body_xs = [x for x, y, c in rotated.get("body", []) if c > 0]
        if body_xs:
            min_bx, max_bx = min(body_xs), max(body_xs)
            if min_bx < 0 and max_bx > w:
                shift_x = 0.0  # figure wider than frame — no good single shift
            elif min_bx < 0:
                shift_x = -min_bx
            elif max_bx > w:
                shift_x = w - max_bx
            else:
                shift_x = 0.0
            if abs(shift_x) > 0.5:
                print(
                    f"[OpenPose Rotator] Image {image_index}: figure shifted "
                    f"{shift_x:+.0f}px horizontally to stay in frame."
                )
                rotated = {
                    part: [(px + shift_x, py, c) for px, py, c in pts]
                    for part, pts in rotated.items()
                }

    # Optional horizontal recentering.
    # Shifts the entire figure so the midpoint between its leftmost and rightmost visible
    # body keypoints aligns with the horizontal centre of the image.
    # Vertical positions are not touched. Applied after all other positioning corrections.
    if recenter:
        body_xs = [x for x, y, c in rotated.get("body", []) if c > 0]
        if body_xs:
            fig_center_x = (min(body_xs) + max(body_xs)) / 2.0
            shift_x = (w / 2.0) - fig_center_x
            if abs(shift_x) > 0.5:
                rotated = {
                    part: [(px + shift_x, py, c) for px, py, c in pts]
                    for part, pts in rotated.items()
                }

    # Build debug axis line (after rotation + anchor positioning)
    axis_line = None
    if debug:
        body_out = rotated.get("body", [])

        if stance == "both_feet":
            # Spine axis: neck → midpoint of both ankles
            neck_out = body_out[1] if len(body_out) > 1 else None
            r_ankle_out = body_out[10] if len(body_out) > 10 else None
            l_ankle_out = body_out[13] if len(body_out) > 13 else None
            if (neck_out is not None and neck_out[2] > 0
                    and r_ankle_out is not None and r_ankle_out[2] > 0
                    and l_ankle_out is not None and l_ankle_out[2] > 0):
                ankle_mid_x = (r_ankle_out[0] + l_ankle_out[0]) / 2
                ankle_mid_y = (r_ankle_out[1] + l_ankle_out[1]) / 2
                axis_line = (
                    (int(round(neck_out[0])), int(round(neck_out[1]))),
                    (int(round(ankle_mid_x)), int(round(ankle_mid_y))),
                )
                print(
                    f"[OpenPose Rotator] Image {image_index}: both feet — spine axis "
                    f"from neck ({neck_out[0]:.0f},{neck_out[1]:.0f}) "
                    f"to ankle mid ({ankle_mid_x:.0f},{ankle_mid_y:.0f})."
                )

        elif stance == "one_foot" and ground_ankle_idx is not None:
            # Vertical axis overlapping the ground ankle node
            ank_out = body_out[ground_ankle_idx] if ground_ankle_idx < len(body_out) else None
            if ank_out is not None and ank_out[2] > 0:
                ax = int(round(ank_out[0]))
                ay = int(round(ank_out[1]))
                axis_line = ((ax, 0), (ax, h))
                print(
                    f"[OpenPose Rotator] Image {image_index}: one foot — vertical axis "
                    f"at ankle ({ax}, {ay})."
                )

        if axis_line is None and orig_anchor is not None and rot_anchor is not None:
            # Fallback (stance unknown or required keypoints missing)
            dx_fb = orig_anchor[0] - rot_anchor[0]
            axis_xi = int(round(pivot[0] + dx_fb))
            axis_line = ((axis_xi, 0), (axis_xi, h))

    # Console: post-rotated figure (after rotation and anchor positioning) - only when debug
    if debug:
        post_dict = _keypoints_to_openpose_dict(rotated)
        print(f"[OpenPose Rotator] Image {image_index} POST-ROTATED (direction={direction}, degrees={degrees}):")
        print(f"  body: {rotated.get('body', [])}")
        print(f"  pose_keypoints_2d (flat): {post_dict.get('people', [{}])[0].get('pose_keypoints_2d', [])}")

    # Render using static connection and draw-order definitions
    rendered = render_openpose_image(rotated, w, h, debug=debug, axis_line=axis_line)
    return rendered, True, rotated
