"""
OpenPose keypoint parsing, torso detection, 3D rotation, and skeleton rendering.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# Body keypoint indices (0-based, COCO format)
# 0=nose, 1=neck, 2=r_shoulder, 3=r_elbow, 4=r_wrist, 5=l_shoulder, 6=l_elbow, 7=l_wrist,
# 8=mid_hip, 9=r_hip, 10=r_knee, 11=r_ankle, 12=l_hip, 13=l_knee, 14=l_ankle,
# 15=r_eye, 16=l_eye, 17=r_ear, 18=l_ear
TORSO_INDICES = [1, 2, 5, 8, 9, 12]  # neck, shoulders, mid_hip, hips

# Limb connections for body (1-indexed in ControlNet util, we use 0-indexed)
# limbSeq: (from_part, to_part) - 0-indexed
BODY_LIMBS = [
    (1, 2),   # neck - r_shoulder
    (1, 5),   # neck - l_shoulder
    (2, 3),   # r_shoulder - r_elbow
    (3, 4),   # r_elbow - r_wrist
    (5, 6),   # l_shoulder - l_elbow
    (6, 7),   # l_elbow - l_wrist
    (1, 8),   # neck - mid_hip
    (8, 9),   # mid_hip - r_hip
    (9, 10),  # r_hip - r_knee
    (10, 11), # r_knee - r_ankle
    (8, 12),  # mid_hip - l_hip
    (12, 13), # l_hip - l_knee
    (13, 14), # l_knee - l_ankle
    (0, 1),   # nose - neck
    (0, 15),  # nose - r_eye
    (15, 17), # r_eye - r_ear
    (0, 16),  # nose - l_eye
    (16, 18), # l_eye - l_ear
    (2, 17),  # r_shoulder - r_ear
    (5, 18),  # l_shoulder - l_ear
]

# Hand edges (0-indexed, 21 keypoints per hand)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Colors for body limbs (BGR for OpenCV)
BODY_COLORS = [
    (0, 0, 255), (0, 85, 255), (0, 170, 255), (0, 255, 255), (0, 255, 170), (0, 255, 85),
    (0, 255, 0), (85, 255, 0), (170, 255, 0), (255, 255, 0), (255, 170, 0), (255, 85, 0),
    (255, 0, 0), (255, 0, 85), (255, 0, 170), (255, 0, 255), (170, 0, 255), (85, 0, 255),
]


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
    Returns (cx, cy) or None if insufficient keypoints.
    """
    body = keypoints.get("body", [])
    if len(body) < 2:
        return None

    xs, ys, total_conf = 0.0, 0.0, 0.0
    for idx in TORSO_INDICES:
        if idx >= len(body):
            continue
        x, y, c = body[idx]
        if c > 0:
            xs += x * c
            ys += y * c
            total_conf += c

    if total_conf <= 0:
        return None

    return (xs / total_conf, ys / total_conf)


def rotate_keypoints_3d(
    keypoints: dict[str, list[tuple[float, float, float]]],
    pivot: tuple[float, float],
    degrees: float,
    direction: str,
    depth_scale: float = 0.4,
) -> dict[str, list[tuple[float, float, float]]]:
    """
    Rotate all keypoints around Y-axis through pivot. Left = positive angle (CCW), right = negative.
    Uses z = (x - pivot_x) * depth_scale for depth inference.
    """
    theta_deg = degrees if direction == "left" else -degrees
    theta = math.radians(theta_deg)
    cx, cy = pivot
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    def rotate_point(x: float, y: float) -> tuple[float, float]:
        z = (x - cx) * depth_scale
        x_new = (x - cx) * cos_t - z * sin_t + cx
        y_new = y  # Y unchanged for Y-axis rotation
        return (x_new, y_new)

    result: dict[str, list[tuple[float, float, float]]] = {}
    for part, points in keypoints.items():
        rotated = []
        for x, y, c in points:
            if c > 0:
                nx, ny = rotate_point(x, y)
                rotated.append((nx, ny, c))
            else:
                rotated.append((x, y, c))
        result[part] = rotated

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
    Draw OpenPose skeleton on blank canvas. White background, colored limbs and joints.
    """
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    body = keypoints.get("body", [])

    # Draw body limbs
    for i, (a, b) in enumerate(BODY_LIMBS):
        if a >= len(body) or b >= len(body):
            continue
        x1, y1, c1 = body[a]
        x2, y2, c2 = body[b]
        if c1 > 0 and c2 > 0:
            color = BODY_COLORS[i % len(BODY_COLORS)]
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

    # Draw body joints
    for i, (x, y, c) in enumerate(body):
        if c > 0:
            color = BODY_COLORS[i % len(BODY_COLORS)]
            cv2.circle(canvas, (int(x), int(y)), 4, color, -1)

    # Draw hands
    for hand_pts in [keypoints.get("hand_left", []), keypoints.get("hand_right", [])]:
        if len(hand_pts) < 2:
            continue
        for a, b in HAND_EDGES:
            if a >= len(hand_pts) or b >= len(hand_pts):
                continue
            x1, y1, c1 = hand_pts[a]
            x2, y2, c2 = hand_pts[b]
            if c1 > 0 and c2 > 0:
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        for x, y, c in hand_pts:
            if c > 0:
                cv2.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), -1)

    return canvas


def rotate_openpose(
    image: np.ndarray,
    pose_keypoint: Any | None,
    direction: str,
    degrees: float,
) -> tuple[np.ndarray, bool]:
    """
    Main pipeline: extract/parse keypoints, detect torso, rotate, render.
    Returns (output_image, success). On failure, returns (input_image, False).
    """
    h, w = image.shape[:2]

    # Get keypoints
    if pose_keypoint is not None:
        keypoints = parse_pose_keypoint(pose_keypoint)
    else:
        keypoints = extract_keypoints_from_image(image)

    if keypoints is None:
        return image, False

    # Torso pivot
    pivot = compute_torso_pivot(keypoints)
    if pivot is None:
        return image, False

    # Rotate
    rotated = rotate_keypoints_3d(keypoints, pivot, degrees, direction)
    rotated = join_broken_segments(rotated, 0.0)

    # Render
    rendered = render_openpose_image(rotated, w, h)
    return rendered, True
