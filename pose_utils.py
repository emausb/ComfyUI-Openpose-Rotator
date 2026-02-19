"""
OpenPose keypoint parsing, torso detection, 3D rotation, and skeleton rendering.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# Body keypoint indices (0-based, OpenPose COCO format)
# 0=nose, 1=neck, 2=r_shoulder, 3=r_elbow, 4=r_wrist, 5=l_shoulder, 6=l_elbow, 7=l_wrist,
# 8=mid_hip, 9=r_hip, 10=r_knee, 11=r_ankle, 12=l_hip, 13=l_knee, 14=l_ankle,
# 15=r_eye, 16=l_eye, 17=r_ear, 18=l_ear
TORSO_INDICES = [1, 2, 5, 8, 9, 12]  # neck, shoulders, mid_hip, hips

# Face/head keypoints (hidden when rotated >90° - back of head visible)
FACE_INDICES = {0, 15, 16, 17, 18}

# Limb-specific depth scales (OpenPose COCO body indices 0-18)
# Torso compact; arms/legs extend further; head moderate
# Based on typical human proportions in frontal view
BODY_DEPTH_SCALES: dict[int, float] = {
    0: 0.30,   # nose - head
    1: 0.20,   # neck - torso center
    2: 0.35,   # r_shoulder
    3: 0.50,   # r_elbow - arm extends
    4: 0.55,   # r_wrist
    5: 0.35,   # l_shoulder
    6: 0.50,   # l_elbow
    7: 0.55,   # l_wrist
    8: 0.22,   # mid_hip - torso
    9: 0.30,   # r_hip
    10: 0.45,  # r_knee - leg extends
    11: 0.50,  # r_ankle
    12: 0.30,  # l_hip
    13: 0.45,  # l_knee
    14: 0.50,  # l_ankle
    15: 0.28,  # r_eye
    16: 0.28,  # l_eye
    17: 0.30,  # r_ear
    18: 0.30,  # l_ear
}

# Default depth scale for hands (21 keypoints each - no per-index in OpenPose hand spec)
HAND_DEPTH_SCALE = 0.45

# Static limb definitions: (a, b) connection, draw layer (0=head first, 4=feet last), color index
# Connections never change; draw order is anatomical (head -> torso -> arms -> legs)
BODY_LIMBS = [
    ((0, 1), 0, 0),   # nose - neck (head)
    ((0, 15), 0, 0),  # nose - r_eye
    ((0, 16), 0, 0),  # nose - l_eye
    ((15, 17), 0, 0), # r_eye - r_ear
    ((16, 18), 0, 0),# l_eye - l_ear
    ((1, 2), 0, 0),   # neck - r_shoulder
    ((1, 5), 0, 0),   # neck - l_shoulder
    ((2, 3), 1, 2),   # r_shoulder - r_elbow
    ((3, 4), 2, 3),   # r_elbow - r_wrist
    ((5, 6), 1, 4),   # l_shoulder - l_elbow
    ((6, 7), 2, 5),   # l_elbow - l_wrist
    ((1, 8), 0, 6),   # neck - mid_hip (spine)
    ((8, 9), 3, 7),   # mid_hip - r_hip
    ((9, 10), 3, 8),  # r_hip - r_knee
    ((10, 11), 4, 9), # r_knee - r_ankle
    ((8, 12), 3, 10), # mid_hip - l_hip
    ((12, 13), 3, 11),# l_hip - l_knee
    ((13, 14), 4, 12),# l_knee - l_ankle
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

# Static joint definitions: index -> (draw layer, color index)
# Draw order is anatomical (head -> torso -> arms -> legs)
JOINT_LAYERS = {
    0: (0, 0),   # nose
    1: (0, 0), 2: (1, 2), 3: (1, 2), 4: (1, 3), 5: (1, 4), 6: (1, 4), 7: (1, 5), 8: (0, 6),
    9: (2, 7), 10: (2, 8), 11: (3, 9), 12: (2, 10), 13: (2, 11), 14: (3, 12),
    15: (0, 0), 16: (0, 0), 17: (0, 0), 18: (0, 0),  # r_eye, l_eye, r_ear, l_ear
}

# Valid keypoint indices for limb drawing (body 0-18, COCO format)
MAIN_BODY_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}

# Hand edges (0-indexed, 21 keypoints per hand) - used when hands enabled
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
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

    # Fallback for "no torso segment" / stick-figure variants: derive missing points
    # Neck (1) from shoulders (2, 5); mid_hip (8) from hips (9, 12)
    if get_pt(1) is None:
        r_sh = get_pt(2)
        l_sh = get_pt(5)
        if r_sh is not None and l_sh is not None:
            nx = (r_sh[0] + l_sh[0]) / 2
            ny = (r_sh[1] + l_sh[1]) / 2
            xs += nx
            ys += ny
            total_conf += 1.0
    if get_pt(8) is None:
        r_hip = get_pt(9)
        l_hip = get_pt(12)
        if r_hip is not None and l_hip is not None:
            mx = (r_hip[0] + l_hip[0]) / 2
            my = (r_hip[1] + l_hip[1]) / 2
            xs += mx
            ys += my
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


def rotate_keypoints_3d(
    keypoints: dict[str, list[tuple[float, float, float]]],
    pivot: tuple[float, float],
    degrees: float,
    direction: str,
    depth_scale: float | None = None,
) -> dict[str, list[tuple[float, float, float]]]:
    """
    Rotate all keypoints around Y-axis through pivot. Left = positive angle (CCW), right = negative.
    Uses limb-specific depth inference per OpenPose COCO body indices for 3D-style projection.
    """
    theta_deg = degrees if direction == "left" else -degrees
    theta = math.radians(theta_deg)
    cx, cy = pivot

    if depth_scale is None:
        depth_scale = _compute_adaptive_depth_scale(keypoints, pivot)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    def rotate_point(
        x: float, y: float, scale: float
    ) -> tuple[float, float]:
        # Side-aware: (x - cx) gives correct sign for left/right body sides
        z = (x - cx) * scale
        x_new = (x - cx) * cos_t - z * sin_t + cx
        y_new = y
        return (x_new, y_new)

    result: dict[str, list[tuple[float, float, float]]] = {}

    for part, points in keypoints.items():
        rotated: list[tuple[float, float, float]] = []

        for i, (x, y, c) in enumerate(points):
            if c <= 0:
                rotated.append((x, y, c))
                continue

            scale = BODY_DEPTH_SCALES.get(i, HAND_DEPTH_SCALE) if part == "body" else HAND_DEPTH_SCALE
            nx, ny = rotate_point(x, y, depth_scale * scale)
            rotated.append((nx, ny, c))

        result[part] = rotated

    # Face visibility: when rotated >90°, hide face/head keypoints (back of head)
    if abs(theta_deg) >= 90:
        if "body" in result:
            body = result["body"]
            for idx in FACE_INDICES:
                if idx < len(body):
                    x, y, c = body[idx]
                    body[idx] = (x, y, 0.0)
        # Also hide separate face keypoints (OpenPose face mesh)
        if "face" in result:
            result["face"] = [(x, y, 0.0) for x, y, c in result["face"]]

    return result


def _fit_and_center_keypoints_on_canvas(
    keypoints: dict[str, list[tuple[float, float, float]]],
    width: int,
    height: int,
    padding: float = 0.05,
) -> dict[str, list[tuple[float, float, float]]]:
    """
    Scale keypoints to fit within canvas (if they exceed bounds), then center.
    Prevents feet/head from being cut off when the figure is taller than the image.
    """
    all_x, all_y, total_conf = 0.0, 0.0, 0.0
    for pts in keypoints.values():
        for x, y, c in pts:
            if c > 0:
                all_x += x * c
                all_y += y * c
                total_conf += c

    if total_conf <= 0:
        return keypoints

    # Bounding box of visible keypoints
    xs = [x for pts in keypoints.values() for x, y, c in pts if c > 0]
    ys = [y for pts in keypoints.values() for x, y, c in pts if c > 0]
    if not xs or not ys:
        return keypoints

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Centroid (stays fixed when scaling around it)
    cx = all_x / total_conf
    cy = all_y / total_conf

    # Max extent from centroid in each direction (we center the centroid, so we must fit
    # the larger of the two sides; using bbox_h/2 fails when centroid is off-center)
    extent_x = max(cx - min_x, max_x - cx)
    extent_y = max(cy - min_y, max_y - cy)

    # Usable canvas with padding (half-extent each side of center)
    pad_w = width * padding
    pad_h = height * padding
    usable_half_w = (width - 2 * pad_w) / 2.0
    usable_half_h = (height - 2 * pad_h) / 2.0

    # Scale down so extent from centroid fits within usable half-canvas
    scale = 1.0
    if extent_x > usable_half_w and usable_half_w > 0:
        scale = min(scale, usable_half_w / extent_x)
    if extent_y > usable_half_h and usable_half_h > 0:
        scale = min(scale, usable_half_h / extent_y)
    target_x = width / 2.0
    target_y = height / 2.0
    dx = target_x - cx
    dy = target_y - cy

    result: dict[str, list[tuple[float, float, float]]] = {}
    for part, pts in keypoints.items():
        result[part] = [
            (cx + (x - cx) * scale + dx, cy + (y - cy) * scale + dy, c)
            for x, y, c in pts
        ]
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


def render_openpose_image(
    keypoints: dict[str, list[tuple[float, float, float]]],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Draw OpenPose skeleton on blank canvas. Black background (matches ControlNet pose images).
    Uses static connection and draw-order definitions; no depth computation.
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

    # Hands and face rendering - commented out to focus on main body composition
    # for hand_key, hand_pts in [("hand_left", keypoints.get("hand_left", [])),
    #                            ("hand_right", keypoints.get("hand_right", []))]:
    #     if len(hand_pts) < 2:
    #         continue
    #     ...
    #     cv2.line(canvas, ...)
    #     cv2.circle(canvas, ...)

    return canvas


def rotate_openpose(
    image: np.ndarray,
    pose_keypoint: Any | None,
    direction: str,
    degrees: float,
    image_index: int = 0,
) -> tuple[np.ndarray, bool, dict[str, list[tuple[float, float, float]]] | None]:
    """
    Main pipeline: extract/parse keypoints, detect torso, rotate, render.
    Returns (output_image, success, rotated_keypoints). On failure, returns (input_image, False, None).
    rotated_keypoints is internal format: {"body": [(x,y,c),...], "hand_left": [...], ...}
    image_index: used for console logging when processing batches.
    """
    h, w = image.shape[:2]

    # Get keypoints
    if pose_keypoint is not None:
        keypoints = parse_pose_keypoint(pose_keypoint)
    else:
        keypoints = extract_keypoints_from_image(image)

    if keypoints is None:
        return image, False, None

    # Scale normalized coords to image size (ComfyUI may pass 0-1 range)
    keypoints = _scale_keypoints_to_image(keypoints, w, h)

    # Console: pre-rotated figure (before rotation, after scaling)
    pre_dict = _keypoints_to_openpose_dict(keypoints)
    print(f"[OpenPose Rotator] Image {image_index} PRE-ROTATED (direction={direction}, degrees={degrees}):")
    print(f"  body: {keypoints.get('body', [])}")
    print(f"  pose_keypoints_2d (flat): {pre_dict.get('people', [{}])[0].get('pose_keypoints_2d', [])}")

    # Torso pivot
    pivot = compute_torso_pivot(keypoints)
    if pivot is None:
        return image, False, None

    # Rotate keypoints around torso pivot
    rotated = rotate_keypoints_3d(keypoints, pivot, degrees, direction)
    rotated = join_broken_segments(rotated, 0.0)

    # Scale to fit and center figure (prevents feet/head cutoff, fixes right/left bias)
    rotated = _fit_and_center_keypoints_on_canvas(rotated, w, h)

    # Console: post-rotated figure (after rotation, fit, and center)
    post_dict = _keypoints_to_openpose_dict(rotated)
    print(f"[OpenPose Rotator] Image {image_index} POST-ROTATED (direction={direction}, degrees={degrees}):")
    print(f"  body: {rotated.get('body', [])}")
    print(f"  pose_keypoints_2d (flat): {post_dict.get('people', [{}])[0].get('pose_keypoints_2d', [])}")

    # Render using static connection and draw-order definitions
    rendered = render_openpose_image(rotated, w, h)
    return rendered, True, rotated
