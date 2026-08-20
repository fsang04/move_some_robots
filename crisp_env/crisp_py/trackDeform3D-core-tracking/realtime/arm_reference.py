"""Rendered-arm depth reference for the live tracking port.

Builds, per frame, the depth image the camera WOULD see of the two Franka
arms alone, from their live joint configurations. armdiff_segmenter.py
subtracts this from the measured depth, which is the faithful live port of
the offline deformable_seg method (there the arm reference is a synchronized
arm-only REPLAY of the same trajectory; here it is rendered from kinematics,
as README.md §4 already proposed).

No pyrender / OpenGL / xacro / yourdfpy: the franka_description xacro keeps
all joint origins in robots/<arm>/kinematics.yaml (every joint is a rotation
about the local z axis), so FK is a 10-line matrix chain, and the collision
meshes are plain binary STLs that 30 lines of numpy can read. Rendering is
point-based: each link's surface is sampled ONCE at start-up (area-weighted),
per frame the points are transformed by FK + the hand-eye calibration and
z-buffered into the pixel grid with np.minimum.at, then a min-filter
(cv2.erode) closes the pinholes between samples. numpy + cv2 only, ~3-6 ms
per frame for two arms at 720p.

Conventions (must match the rest of the pipeline):
  * kinematics.yaml, STL meshes, T_base2cam are METERS; the returned depth
    image is uint16-style float32 MILLIMETERS, 0 = no arm (same convention
    as the Kinect depth frames).
  * T_base2cam maps robot-base coordinates to OpenCV camera coordinates
    (x right, y down, z forward) -- the same `T_left_base2cam` /
    `T_right_base2cam` that utils/transforms.py loads.
  * The projection uses the pinhole K with no distortion. KinectSource
    undistorts with newCameraMatrix = K (frame_source.py:112), so this
    matches ONLY when the hand-eye calibration was made under that same
    convention (README.md §4, REALTIME_SAM2_OVERVIEW.md §7.1).
"""
import re
import struct
from pathlib import Path

import cv2
import numpy as np

# franka_description, relative to this file (realtime/ -> trackDeform3D -> crisp_py)
DEFAULT_DESC_ROOT = (Path(__file__).resolve().parents[2]
                     / 'external_pkgs' / 'franka_description'
                     / 'share' / 'franka_description')

# Fixed transforms the xacro hard-codes instead of putting in kinematics.yaml:
FLANGE_Z = 0.107          # joint8: link7 -> link8/flange   (franka_arm.xacro)
HAND_YAW = -np.pi / 4     # flange -> hand                  (franka_hand_arguments.xacro)
FINGER_Z = 0.0584         # hand -> finger joints           (common/franka_hand.xacro)
FINGER_OPEN = 0.04        # per-finger travel limit, m
GRASP_Z = 0.1034          # offset from hand -> grasp point, along hand +z


# ============================================================================
# small geometry helpers
# ============================================================================

def rpy_matrix(roll, pitch, yaw):
    """URDF rpy -> 3x3 rotation (extrinsic x-y-z: R = Rz @ Ry @ Rx)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def rot_z(q):
    c, s = np.cos(q), np.sin(q)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def load_kinematics(yaml_path):
    """robots/<arm>/kinematics.yaml -> list of 4x4 joint-origin transforms
    for joint1..joint8 (joint8 = flange). The file is a fixed two-level
    mapping of scalars, so a 12-line parser reads it and PyYAML is not a
    dependency of the tracking environment."""
    values = {}                                   # {'joint1': {'x': 0., ...}}
    current = None
    for line in Path(yaml_path).read_text().splitlines():
        m = re.match(r'^(joint\d+):', line)
        if m:
            current = values.setdefault(m.group(1), {})
            continue
        m = re.match(r'^\s+(x|y|z|roll|pitch|yaw):\s*([-+0-9.eE]+)', line)
        if m and current is not None:
            current[m.group(1)] = float(m.group(2))
    origins = []
    for i in range(1, 9):
        k = values[f'joint{i}']
        origins.append(transform((k.get('x', 0), k.get('y', 0), k.get('z', 0)),
                                 (k.get('roll', 0), k.get('pitch', 0), k.get('yaw', 0))))
    return origins


def load_stl(path):
    """Binary STL -> (n_tri, 3, 3) float64 vertex array (meters)."""
    raw = Path(path).read_bytes()
    if raw[:5].lower() == b'solid' and b'facet' in raw[:500]:
        raise ValueError(f'{path} looks like ASCII STL; only binary is supported')
    n = struct.unpack('<I', raw[80:84])[0]
    if len(raw) < 84 + n * 50:
        raise ValueError(f'{path}: truncated binary STL')
    rec = np.frombuffer(raw, dtype=np.dtype([('normal', '<f4', (3,)),
                                             ('verts', '<f4', (3, 3)),
                                             ('attr', '<u2')]),
                        count=n, offset=84)
    return rec['verts'].astype(np.float64)


def box_tris(size, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    """Axis-aligned box (URDF <box size>) posed by xyz/rpy -> (12, 3, 3) tris."""
    sx, sy, sz = np.asarray(size) / 2.0
    corners = np.array([[x, y, z] for x in (-sx, sx)
                        for y in (-sy, sy) for z in (-sz, sz)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = [idx for q in quads for idx in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    tris = corners[np.asarray(faces)]
    R = rpy_matrix(*rpy)
    return tris @ R.T + np.asarray(xyz)


def finger_collision_tris():
    """The franka hand FINGER as franka_description's four collision boxes
    (common/franka_hand.xacro:37-64) -- there is no collision STL for it."""
    return np.concatenate([
        box_tris((22e-3, 15e-3, 20e-3), (0, 18.5e-3, 11e-3)),          # screw mount
        box_tris((22e-3, 8.8e-3, 3.8e-3), (0, 6.8e-3, 2.2e-3)),        # sledge
        box_tris((17.5e-3, 7e-3, 23.5e-3), (0, 15.9e-3, 28.35e-3),
                 (np.pi / 6, 0, 0)),                                   # diagonal finger
        box_tris((17.5e-3, 15.2e-3, 18.5e-3), (0, 7.58e-3, 45.25e-3)),  # rubber tip
    ])


def sample_surface(tris, n_points, rng):
    """Area-weighted random points on a triangle soup. (n_points, 3)."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = areas.sum()
    if total <= 0:
        raise ValueError('degenerate mesh: zero surface area')
    idx = rng.choice(len(tris), size=n_points, p=areas / total)
    u = rng.random(n_points)
    v = rng.random(n_points)
    flip = u + v > 1.0                       # fold back into the triangle
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return (v0[idx] + u[:, None] * (v1[idx] - v0[idx])
            + v[:, None] * (v2[idx] - v0[idx]))


# ============================================================================
# FK + sampled surface of one Franka arm (fr3 / fer / fp3 + franka hand)
# ============================================================================

class FrankaArmModel:
    """One arm's kinematics + surface point samples, all in the BASE frame.

    Frames follow franka_description exactly: link_i sits after joint_i
    (all revolute about local z, origins from kinematics.yaml), the flange
    after the fixed joint8, the hand yawed -45 deg on the flange, and the two
    fingers prismatic along +/-y of the hand at z=0.0584 (the right finger
    mesh is the left one yawed 180 deg).

    n_points is the TOTAL sample budget for the arm, split across the link,
    hand, and finger meshes by surface area. The collision meshes are used:
    they are tight, watertight, load as plain binary STL, and the segmenter
    dilates its arm mask anyway.
    """

    def __init__(self, desc_root=DEFAULT_DESC_ROOT, arm: str = 'fr3',
                 hand: str = 'franka_hand_white', n_points: int = 20000,
                 seed: int = 0):
        desc_root = Path(desc_root)
        self.joint_origins = load_kinematics(desc_root / 'robots' / arm / 'kinematics.yaml')

        arm_dir = desc_root / 'meshes' / 'robot_arms' / arm / 'collision'
        meshes = [load_stl(arm_dir / f'link{i}.stl') for i in range(8)]
        if hand:
            hand_dir = desc_root / 'meshes' / 'robot_ee' / hand / 'collision'
            finger = finger_collision_tris()     # no finger STL ships; boxes do
            meshes += [load_stl(hand_dir / 'hand.stl'), finger, finger]
        self.has_hand = bool(hand)

        rng = np.random.default_rng(seed)
        areas = np.array([0.5 * np.linalg.norm(
            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()
            for t in meshes])
        counts = np.maximum(50, (n_points * areas / areas.sum()).astype(int))
        # local-frame samples, homogeneous, one (n_i, 4) array per body
        self.samples = []
        for tris, cnt in zip(meshes, counts):
            pts = sample_surface(tris, int(cnt), rng)
            self.samples.append(np.hstack([pts, np.ones((len(pts), 1))]))

    def body_transforms(self, q, finger_q: float = FINGER_OPEN):
        """7 joint angles (rad) -> list of 4x4 base-frame transforms, one per
        sampled body: link0..link7 [, hand, leftfinger, rightfinger].
        finger_q = per-finger travel in m (0 = closed .. 0.04 = open)."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(f'expected 7 joint angles, got shape {q.shape}')
        T = np.eye(4)
        frames = [T]                                        # link0
        for i in range(7):
            T = T @ self.joint_origins[i] @ rot_z(q[i])
            frames.append(T)                                # link1..link7
        if self.has_hand:
            w = float(np.clip(finger_q, 0.0, FINGER_OPEN))
            T_hand = T @ self.joint_origins[7] @ rot_z(HAND_YAW)
            frames.append(T_hand)
            frames.append(T_hand @ transform((0.0, w, FINGER_Z)))
            frames.append(T_hand @ transform((0.0, -w, FINGER_Z), (0, 0, np.pi)))
        return frames

    def points_base(self, q, finger_q: float = FINGER_OPEN):
        """(N, 3) surface points in the robot BASE frame, meters."""
        frames = self.body_transforms(q, finger_q)
        return np.concatenate([(pts @ T.T)[:, :3]
                               for pts, T in zip(self.samples, frames)])

    def grasp_point_base(self, q, grasp_z: float = GRASP_Z):
        """(3,) the point BETWEEN the fingertips, robot base frame, meters.

        This point is the EE position the tracker wants: the gripper holds the
        cable there, not at the hand origin. grasp_z is the offset along the
        hand +z axis, and it does not depend on the finger travel (the two
        fingers close symmetrically about that axis)."""
        if not self.has_hand:
            raise RuntimeError('grasp_point_base() needs a hand model; this '
                               'FrankaArmModel was built with hand=None')
        T_hand = self.body_transforms(q)[8]     # link0..link7 = 0..7, hand = 8
        return T_hand[:3, 3] + T_hand[:3, 2] * float(grasp_z)


# ============================================================================
# per-frame depth renderer (any number of arms, one camera)
# ============================================================================

class ArmDepthRenderer:
    """Splats the arms' sampled surfaces into a camera depth image.

    arms: {name: (FrankaArmModel, T_base2cam 4x4 in meters)} -- the SAME
    transforms utils/transforms.py loads from transform_ee_cam_world.npz.

    render() -> (H, W) float32 depth in mm, 0 = no arm. The min-filter
    (splat_ksize) closes the gaps between point samples; it also grows the
    silhouette by ~splat_ksize/2 px, which is the conservative direction for
    background subtraction. z_near_mm drops points behind / grazing the
    camera before projection.
    """

    _NO_HIT = np.float32(1e9)

    def __init__(self, arms: dict, K, shape, splat_ksize: int = 5,
                 z_near_mm: float = 150.0):
        self.arms = dict(arms)
        self.K = np.asarray(K, dtype=np.float64)
        self.H, self.W = int(shape[0]), int(shape[1])
        self.splat_ksize = int(splat_ksize)
        self.z_near_mm = float(z_near_mm)
        self._kernel = np.ones((self.splat_ksize, self.splat_ksize), np.uint8)

    def render(self, q_by_arm: dict, finger_q_by_arm: dict = None) -> np.ndarray:
        """q_by_arm: {name: (7,) rad}; finger_q_by_arm: {name: m} (default
        fully open -- the largest silhouette, i.e. the conservative one)."""
        pts_cam = []
        for name, (model, T_base2cam) in self.arms.items():
            q = q_by_arm[name]
            fq = FINGER_OPEN if finger_q_by_arm is None \
                else finger_q_by_arm.get(name, FINGER_OPEN)
            p = model.points_base(q, fq)                     # (N,3) base, m
            p = p @ T_base2cam[:3, :3].T + T_base2cam[:3, 3]
            pts_cam.append(p * 1000.0)                       # -> camera mm
        p = np.concatenate(pts_cam)

        z = p[:, 2]
        ok = z > self.z_near_mm
        u = np.rint(p[ok, 0] * self.K[0, 0] / z[ok] + self.K[0, 2]).astype(np.int64)
        v = np.rint(p[ok, 1] * self.K[1, 1] / z[ok] + self.K[1, 2]).astype(np.int64)
        z = z[ok].astype(np.float32)
        ok = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)

        buf = np.full(self.H * self.W, self._NO_HIT, dtype=np.float32)
        np.minimum.at(buf, v[ok] * self.W + u[ok], z[ok])
        buf = buf.reshape(self.H, self.W)
        if self.splat_ksize > 1:
            buf = cv2.erode(buf, self._kernel)               # min-filter spread
        buf[buf >= self._NO_HIT * 0.5] = 0.0
        return buf

    def grasp_points_cam_mm(self, q_by_arm: dict, grasp_z: float = GRASP_Z) -> dict:
        """{name: (3,) grasp point in CAMERA-frame mm}, from the joints alone.

        The EE source for an armdiff session: FK plus the same hand-eye
        calibration the renderer already uses give both gripper positions
        directly, so the first frame needs no candidate mask, no skeleton and
        no endpoint guess (realtime/bootstrap.py estimates them from an image
        instead, which a wrong candidate can hijack).
        """
        out = {}
        for name, (model, T_base2cam) in self.arms.items():
            p = model.grasp_point_base(q_by_arm[name], grasp_z)
            out[name] = (T_base2cam[:3, :3] @ p + T_base2cam[:3, 3]) * 1000.0
        return out


def load_base2cam(calib_npz):
    """transform_ee_cam_world.npz -> {'left': T, 'right': T} (meters), the
    calibration convention utils/transforms.py:8 established."""
    tf = np.load(calib_npz)
    return {'left': np.asarray(tf['T_left_base2cam'], dtype=np.float64),
            'right': np.asarray(tf['T_right_base2cam'], dtype=np.float64)}
