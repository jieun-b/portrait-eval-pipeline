import cv2
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte

LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float32)

IDX_POINTS = [30, 8, 36, 45, 48, 54]


def eye_aspect_ratio(pts, idx):
    A = np.linalg.norm(pts[idx[1]] - pts[idx[5]])
    B = np.linalg.norm(pts[idx[2]] - pts[idx[4]])
    C = np.linalg.norm(pts[idx[0]] - pts[idx[3]])
    return (A + B) / (2.0 * C)


def is_eye_closed(pts, th=0.2):
    return (eye_aspect_ratio(pts, LEFT_EYE) < th or 
            eye_aspect_ratio(pts, RIGHT_EYE) < th)


def extract_landmark(frame, fa):
    img = img_as_ubyte(resize(frame, (256, 256)))
    try:
        lmk = fa.get_landmarks_from_image(img, return_landmark_score=False)
        if lmk is None or len(lmk) == 0:
            return None
        pts = lmk[0]
        return None if is_eye_closed(pts) else pts
    except Exception:
        return None


def compute_pose(landmarks, img_size=(256, 256)):
    if landmarks is None or len(landmarks) < max(IDX_POINTS) + 1:
        return None

    image_points = np.array([landmarks[i] for i in IDX_POINTS], dtype=np.float32)
    focal_length = img_size[1]
    center = (img_size[1] / 2, img_size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
    return yaw, pitch, roll


def is_valid_source(lmk, yaw_th=30, pitch_th=30):
    if lmk is None or is_eye_closed(lmk):
        return False
    pose = compute_pose(lmk)
    if pose is None:
        return False
    yaw, pitch, _ = pose
    return abs(yaw) <= yaw_th and abs(pitch) <= pitch_th
