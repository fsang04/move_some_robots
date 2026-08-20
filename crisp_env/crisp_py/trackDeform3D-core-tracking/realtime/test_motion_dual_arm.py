#!/usr/bin/env python3
"""Freestyle dual-arm test motion, for exercising the live DLO tracker.

A stand-in for the real manipulation policy: both arms wander through a smooth,
non-repeating trajectory while holding the two ends of a cable, so the tracker
sees a continuously deforming DLO with both arms inside the workspace.

WHAT THIS FEEDS THE TRACKER
---------------------------
Nothing, directly -- and that is the point. The live pipeline does not take
joints from a caller; realtime/joint_source.py:CrispJointSource subscribes to
`/<ns>/joint_states` itself and reads `robot.joint_values` per frame. The franka
driver publishes that at ~1 kHz whether or not anything is moving. So the entire
contract of this script is "move the arms" -- run it in a second terminal
alongside dlo_tracking_live.py and the joint configuration at every step is
already there.

    Terminal A   docker compose up launch_dual_franka      (README_ZED.md §3)
    Terminal B   python dlo_tracking_live.py --source zed --segmenter armdiff ...
    Terminal C   python realtime/test_motion_dual_arm.py

Order matters only in that the robots must be up first; B and C are independent.

The npz this script writes is a RECORD, not a feed: joints, commanded and
measured TCPs, and gripper separation per tick, so a session can be checked
after the fact (did the arms actually go where they were told? did separation
hold?) without re-running hardware.

WHY THE TWO ARMS NEED THE CALIBRATION TO MEASURE ONE DISTANCE
-------------------------------------------------------------
`left.end_effector_pose.position` is in the LEFT base frame and
`right.end_effector_pose.position` is in the RIGHT base frame. Subtracting them
is meaningless -- there is no ROS TF edge between the two robot bases either.
The hand-eye calibration is the only thing that relates them:

    T_right_base -> world  =  inv(T_left_base2cam) @ T_right_base2cam

Under zed_calib_003 that puts the bases 1.2555 m apart, facing each other (the
rotation is a 180 deg yaw). This script therefore plans in a shared "world"
frame -- defined as the LEFT arm's base frame -- as a midpoint plus a separation
vector, so the <= 55 cm gripper spacing is enforced by construction rather than
hoped for. Note the arms are 0.64 m apart at the home pose, i.e. already too far:
the motion has to reach inward, it cannot just wobble around home.

`fr3_hand_tcp` is the commanded Cartesian frame (robot_config.py:91), and it is
the same point arm_reference.grasp_point_base() computes from joints -- GRASP_Z
= 0.1034 m along hand +z. So "distance between the grippers" means one thing
everywhere: here, in the tracker, and in the recorded log. start() cross-checks
that identity against live FK before moving anything.

SAFETY
------
Targets pass through, in order: a per-arm workspace box and reach-radius clamp
(in that arm's OWN base frame), a Cartesian speed limit, and finally the
separation clamp. Separation is enforced LAST and so always holds; if a reach
clamp and the separation clamp ever disagree, separation wins and the reach
overshoot is reported. With the default amplitudes no clamp fires at all --
`--dry_run` proves that with no robots and no camera attached.

    python realtime/test_motion_dual_arm.py --dry_run

Ctrl-C is a clean stop: the arms ease back to the start pose and then home.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# .../crisp_py/trackDeform3D-core-tracking/realtime/this_file.py
_HERE = Path(__file__).resolve()
_TRACKING = _HERE.parents[1]           # trackDeform3D-core-tracking
_CRISP_PY = _HERE.parents[2]           # crisp_py
if str(_TRACKING) not in sys.path:
    sys.path.insert(0, str(_TRACKING))

DEFAULT_CALIB = (_CRISP_PY / 'hand_to_eye_calibration' / 'roahm-deformable-objects'
                 / 'captured_calibration_data' / 'zed_calib_003'
                 / 'transform_ee_cam_world.npz')
DEFAULT_IMPEDANCE = _CRISP_PY / 'config' / 'control' / 'default_cartesian_impedance.yaml'
DEFAULT_OUT = _TRACKING / 'output' / 'dual_arm_test_motion'

# Gripper pointing straight DOWN in its own base frame: hand +z -> base -z, and
# hand +y (the finger closing axis) -> base -y. Both arms use this same matrix in
# their own frames, which -- because the right base is yawed 180 deg -- leaves
# both sets of fingers closing along the world y axis, i.e. across a cable that
# runs along the base-to-base (world x) axis. It is also the orientation the
# working calibration captures used (capture_poses_and_images_for_calibration_*.py).
GRIP_DOWN = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]])


# ============================================================================
# trajectory
# ============================================================================

@dataclass
class MotionParams:
    """Freestyle trajectory shape, in the shared world frame (= left base).

    Deliberately incommensurate frequencies: nothing here shares a period, so
    the pose never exactly repeats over a run and the tracker is not tested
    against the same handful of configurations again and again.

    The amplitudes are chosen so every clamp in plan_step() stays inactive --
    see --dry_run, which reports how close each one came to firing.
    """
    # midpoint of the two grippers: mid0 is filled in from the calibration
    mid0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mid_amp: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.08, 0.05]))
    mid_freq: np.ndarray = field(default_factory=lambda: np.array([0.037, 0.053, 0.071]))
    mid_phase: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.1, 0.4]))

    # Separation LENGTH between the grippers. This also sets how far each arm has
    # to reach: with the bases 1.256 m apart, a separation of L puts each TCP
    # (1.256 - L)/2 forward of its own base, so a SMALLER separation is the
    # expensive one for reach, not a larger one.
    sep_nom: float = 0.46
    sep_amp: float = 0.07          # -> length sweeps 0.39 .. 0.53 m
    sep_freq: float = 0.029
    sep_max: float = 0.55          # the hard ceiling; enforced last, always holds

    # separation DIRECTION: unit vector wobbling about world +x, so the cable
    # swings and tilts instead of just stretching along one fixed axis
    dir_amp: np.ndarray = field(default_factory=lambda: np.array([0.35, 0.20]))
    dir_freq: np.ndarray = field(default_factory=lambda: np.array([0.043, 0.061]))
    dir_phase: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.8]))

    # gentle wrist wobble about the tool's own x/y, keeping the gripper near-down
    tilt_amp: float = np.deg2rad(8.0)
    tilt_freq: np.ndarray = field(default_factory=lambda: np.array([0.031, 0.047]))

    # per-arm safety envelope, in that arm's OWN base frame
    box_lo: np.ndarray = field(default_factory=lambda: np.array([0.25, -0.30, 0.18]))
    box_hi: np.ndarray = field(default_factory=lambda: np.array([0.60, 0.30, 0.62]))
    r_min: float = 0.32
    r_max: float = 0.72            # fr3 reach is ~0.855 m; this leaves margin

    max_speed: float = 0.15        # m/s ceiling on the commanded TCP


def _clamp_arm(p_base, prm: MotionParams):
    """Box + reach-radius clamp in one arm's base frame. Returns (p, fired)."""
    q = np.clip(p_base, prm.box_lo, prm.box_hi)
    fired = not np.allclose(q, p_base)
    r = float(np.linalg.norm(q))
    if r > prm.r_max:
        q, fired = q * (prm.r_max / r), True
    elif 1e-6 < r < prm.r_min:
        q, fired = q * (prm.r_min / r), True
    return q, fired


def _limit_step(prev, target, max_step):
    """Cap how far a commanded target may jump in one control tick."""
    if prev is None:
        return target
    d = target - prev
    n = float(np.linalg.norm(d))
    return target if n <= max_step else prev + d * (max_step / n)


def _enforce_sep(pl, pr, sep_max):
    """Pull both grippers symmetrically toward their midpoint until they are at
    most sep_max apart. Runs LAST, so the constraint is unconditional."""
    d = pr - pl
    L = float(np.linalg.norm(d))
    if L <= sep_max or L < 1e-9:
        return pl, pr, L
    c, h = 0.5 * (pl + pr), 0.5 * sep_max * d / L
    return c - h, c + h, sep_max


def raw_targets_world(t, prm: MotionParams):
    """The unclamped freestyle pose at time t: (p_left, p_right) in world."""
    w = 2.0 * np.pi
    mid = prm.mid0 + prm.mid_amp * np.sin(w * prm.mid_freq * t + prm.mid_phase)
    u = np.array([1.0,
                  prm.dir_amp[0] * np.sin(w * prm.dir_freq[0] * t + prm.dir_phase[0]),
                  prm.dir_amp[1] * np.sin(w * prm.dir_freq[1] * t + prm.dir_phase[1])])
    u /= np.linalg.norm(u)
    L = min(prm.sep_nom + prm.sep_amp * np.sin(w * prm.sep_freq * t), prm.sep_max)
    half = 0.5 * L * u
    return mid - half, mid + half


def tool_rotation(t, prm: MotionParams, yaw: float = 0.0):
    """Gripper orientation in the arm's own base frame at time t."""
    from scipy.spatial.transform import Rotation
    w = 2.0 * np.pi
    a = prm.tilt_amp * np.sin(w * prm.tilt_freq[0] * t)
    b = prm.tilt_amp * np.sin(w * prm.tilt_freq[1] * t + 0.9)
    R = Rotation.from_matrix(GRIP_DOWN)
    if abs(yaw) > 0:
        R = Rotation.from_euler('z', yaw) * R
    return R * Rotation.from_euler('xy', [a, b])       # wobble about the TOOL axes


class DualArmPlanner:
    """Turns time into a safe commanded TCP for each arm, in its own base frame."""

    def __init__(self, T_rb2w, prm: MotionParams, dt: float, yaw_left=0.0, yaw_right=0.0):
        self.T_rb2w = np.asarray(T_rb2w, dtype=np.float64)
        self.T_w2rb = np.linalg.inv(self.T_rb2w)
        self.prm = prm
        self.dt = float(dt)
        self.yaw = {'left': float(yaw_left), 'right': float(yaw_right)}
        self._prev = {'left': None, 'right': None}
        self.stats = {'box_clamped': 0, 'sep_clamped': 0, 'speed_clamped': 0,
                      'max_reach': {'left': 0.0, 'right': 0.0},
                      'max_sep': 0.0, 'min_sep': 1e9}

    def to_world(self, p_right_base):
        return self.T_rb2w[:3, :3] @ p_right_base + self.T_rb2w[:3, 3]

    def to_right_base(self, p_world):
        return self.T_w2rb[:3, :3] @ p_world + self.T_w2rb[:3, 3]

    def plan_step(self, t, record_stats=True):
        """-> {'left': p_in_left_base, 'right': p_in_right_base}, sep_m."""
        prm = self.prm
        pl_w, pr_w = raw_targets_world(t, prm)

        # 1. per-arm envelope, each in its own base frame
        pl_b, fl = _clamp_arm(pl_w, prm)                        # world == left base
        pr_b, fr = _clamp_arm(self.to_right_base(pr_w), prm)
        pl_w, pr_w = pl_b, self.to_world(pr_b)

        # 2. Cartesian speed ceiling
        max_step = prm.max_speed * self.dt
        nl, nr = _limit_step(self._prev['left'], pl_w, max_step), \
            _limit_step(self._prev['right'], pr_w, max_step)
        sped = not (np.allclose(nl, pl_w) and np.allclose(nr, pr_w))
        pl_w, pr_w = nl, nr

        # 3. separation last -- this is the constraint that must not be violated
        pl_w, pr_w, sep = _enforce_sep(pl_w, pr_w, prm.sep_max)
        self._prev['left'], self._prev['right'] = pl_w, pr_w

        if record_stats:
            s = self.stats
            s['box_clamped'] += int(fl or fr)
            s['speed_clamped'] += int(sped)
            s['sep_clamped'] += int(sep >= prm.sep_max - 1e-9)
            s['max_sep'], s['min_sep'] = max(s['max_sep'], sep), min(s['min_sep'], sep)

        out = {'left': pl_w, 'right': self.to_right_base(pr_w)}
        if record_stats:
            for k, v in out.items():
                self.stats['max_reach'][k] = max(self.stats['max_reach'][k],
                                                 float(np.linalg.norm(v)))
        return out, sep

    def pose_step(self, t):
        """-> {'left': Pose, 'right': Pose}, sep_m -- ready for robot.set_target()."""
        from crisp_py.utils.geometry import Pose
        pos, sep = self.plan_step(t)
        return {k: Pose(pos[k], tool_rotation(t, self.prm, self.yaw[k]))
                for k in pos}, sep


# ============================================================================
# geometry / calibration
# ============================================================================

def load_world_transform(calib_npz):
    """-> T_right_base -> world (world = LEFT base frame), 4x4 meters."""
    p = Path(calib_npz)
    if not p.exists():
        raise FileNotFoundError(
            f'calibration not found: {p}\nPass --calib explicitly; README_ZED.md §5 '
            'says the default in dlo_tracking_live.py points at the OLD Azure rig.')
    tf = np.load(p)
    L = np.asarray(tf['T_left_base2cam'], dtype=np.float64)
    R = np.asarray(tf['T_right_base2cam'], dtype=np.float64)
    return np.linalg.inv(L) @ R


def make_params(T_rb2w, args) -> MotionParams:
    """Trajectory params anchored to the ACTUAL base-to-base geometry, so this
    keeps working if the calibration is redone or a robot base is moved."""
    prm = MotionParams()
    t = T_rb2w[:3, 3]
    prm.mid0 = np.array([0.5 * t[0], 0.5 * t[1], args.mid_z])
    prm.sep_nom = args.sep_nom
    prm.sep_amp = args.sep_amp
    prm.sep_max = args.max_sep
    prm.max_speed = args.max_speed
    prm.tilt_amp = np.deg2rad(args.tilt_deg)
    for k in ('mid_freq', 'sep_freq', 'dir_freq', 'tilt_freq'):
        setattr(prm, k, getattr(prm, k) * args.speed)
    if prm.sep_nom + prm.sep_amp > prm.sep_max:
        print(f'  NOTE: sep_nom+sep_amp = {prm.sep_nom + prm.sep_amp:.3f} m exceeds '
              f'--max_sep {prm.sep_max:.3f}; the peaks will be flat-topped by the clamp.')
    return prm


# ============================================================================
# dry run -- no robots, no camera
# ============================================================================

def dry_run(planner: DualArmPlanner, duration: float, dt: float):
    print(f'\n--- dry run: {duration:.0f} s at {1/dt:.0f} Hz, no hardware ---')
    n = int(duration / dt)
    sep, reach_l, reach_r, spd = [], [], [], []
    prev = None
    for i in range(n):
        t = i * dt
        pos, s = planner.plan_step(t)
        sep.append(s)
        reach_l.append(np.linalg.norm(pos['left']))
        reach_r.append(np.linalg.norm(pos['right']))
        w = np.concatenate([pos['left'], planner.to_world(pos['right'])])
        if prev is not None:
            spd.append(np.linalg.norm((w[:3] - prev[:3])) / dt)
            spd.append(np.linalg.norm((w[3:] - prev[3:])) / dt)
        prev = w
    sep, reach_l, reach_r, spd = map(np.asarray, (sep, reach_l, reach_r, spd))
    prm = planner.prm
    print(f'  gripper separation   {sep.min()*100:6.1f} .. {sep.max()*100:6.1f} cm'
          f'   (limit {prm.sep_max*100:.0f} cm)   '
          f'{"OK" if sep.max() <= prm.sep_max + 1e-6 else "VIOLATED"}')
    print(f'  left  reach from base {reach_l.min():.3f} .. {reach_l.max():.3f} m'
          f'   (clamp {prm.r_min:.2f} .. {prm.r_max:.2f})')
    print(f'  right reach from base {reach_r.min():.3f} .. {reach_r.max():.3f} m'
          f'   (clamp {prm.r_min:.2f} .. {prm.r_max:.2f})')
    print(f'  peak TCP speed        {spd.max():.3f} m/s   (limit {prm.max_speed:.2f})')
    print(f'  clamps fired: box {planner.stats["box_clamped"]}, '
          f'speed {planner.stats["speed_clamped"]}, '
          f'separation {planner.stats["sep_clamped"]}  (of {n} ticks)')
    planner._prev = {'left': None, 'right': None}   # else the speed limiter, still
    p0, _ = planner.plan_step(0.0, record_stats=False)   # holding t=duration, skews it
    print(f'  start pose: left {np.round(p0["left"], 4)}  '
          f'right {np.round(p0["right"], 4)}   (each in its own base frame)')
    ok = sep.max() <= prm.sep_max + 1e-6
    print(f'--- dry run {"PASSED" if ok else "FAILED"} ---\n')
    return 0 if ok else 1


def check_fk_agreement(robots, tol_mm=15.0):
    """`fr3_hand_tcp` (what we command) must equal arm_reference's grasp point
    (what the tracker renders). If these disagree the separation we enforce is
    not the separation the tracker sees -- worth catching before moving.

    Measured on this rig at the home pose: 0.0 mm on both arms. The pose
    broadcaster derives fr3_hand_tcp from the same franka_description URDF that
    kinematics.yaml comes from, so there is no nominal-vs-calibrated offset to
    absorb here, and the check runs with the arms at rest so the two topics are
    not sampled at meaningfully different instants either. The 15 mm gate is
    therefore almost pure headroom, sized to still catch the one failure mode
    that is not glaring: a GRASP_Z that does not match the tool actually fitted.
    The glaring ones land far outside it -- fr3_link8 instead of fr3_hand_tcp is
    103 mm, and joint_values reading all zeros (Robot built with a leading slash
    on the namespace) is ~400 mm.
    """
    from realtime.arm_reference import FrankaArmModel
    model = FrankaArmModel(n_points=200)
    print('  FK cross-check (commanded fr3_hand_tcp vs arm_reference grasp point):')
    worst = 0.0
    for name, robot in robots.items():
        meas = robot.end_effector_pose.position
        fk = model.grasp_point_base(robot.joint_values[:7])
        err = float(np.linalg.norm(meas - fk)) * 1000.0
        worst = max(worst, err)
        print(f'    {name:5s} tcp {np.round(meas, 4)}  fk {np.round(fk, 4)}  '
              f'delta {err:5.1f} mm')
    if worst > tol_mm:
        zeros = [n for n, r in robots.items() if np.allclose(r.joint_values[:7], 0.0)]
        hint = (f'joint_values is all zeros for {zeros} -- Robot was built with a '
                'namespace that does not match the driver\'s joint-name prefix '
                '(see the note beside the Robot(...) construction)'
                if zeros else
                'check that target_frame is fr3_hand_tcp and that GRASP_Z matches '
                'your tool')
        raise RuntimeError(
            f'TCP and rendered grasp point disagree by {worst:.1f} mm (> {tol_mm} mm). '
            f'The tracker would then measure a different gripper separation than this '
            f'script enforces: {hint}.')
    return worst


# ============================================================================
# live run
# ============================================================================

def run_live(args, planner: DualArmPlanner):
    from std_msgs.msg import Float64MultiArray
    from crisp_py.robot import Robot

    dt = 1.0 / args.rate
    # No leading slash on the namespace: it resolves the same topics either way,
    # but it also seeds Robot's joint-name prefix (robot.py:72), and '/left_'
    # matches none of the driver's 'left_fr3_jointN' names -- joint_values would
    # silently read all zeros. See the longer note in joint_source.py.
    robots = {'left': Robot(namespace='left', name='test_motion_left'),
              'right': Robot(namespace='right', name='test_motion_right')}
    grip_pubs = {}

    def grip(value, repeat=8):
        """1.0 = open, 0.0 = close. Same topic send_both_arms_home.py uses;
        repeated because the first message can be lost before the controller's
        subscription is matched."""
        msg = Float64MultiArray()
        msg.data = [float(value)]
        for _ in range(repeat):
            for pub in grip_pubs.values():
                pub.publish(msg)
            time.sleep(0.1)

    log = {k: [] for k in ('t', 'wall', 'q_left', 'q_right', 'tcp_meas_left',
                           'tcp_meas_right', 'tcp_cmd_left', 'tcp_cmd_right',
                           'sep_cmd', 'sep_meas')}
    worst_sep = 0.0          # bound before the try: the finally block reports it
    try:
        print('Waiting for both arms...')
        for name, robot in robots.items():
            robot.wait_until_ready(timeout=args.timeout)
            grip_pubs[name] = robot.node.create_publisher(
                Float64MultiArray,
                f'/{name}/gripper/gripper_position_controller/commands', 10)
        print('  both arms ready.')

        check_fk_agreement(robots)

        if not args.no_home:
            print('Homing both arms...')
            for robot in robots.values():
                robot.home()

        print('Switching to cartesian_impedance_controller...')
        for robot in robots.values():
            robot.controller_switcher_client.switch_controller(
                'cartesian_impedance_controller')
            robot.cartesian_controller_parameters_client.load_param_config(
                file_path=str(args.impedance))

        # Move to t=0 of the trajectory, so the loop starts with no jump.
        start, sep0 = planner.pose_step(0.0)
        planner._prev = {'left': None, 'right': None}          # undo the probe tick
        print(f'Moving to start pose (grippers {sep0*100:.1f} cm apart)...')
        for name, robot in robots.items():
            robot.move_to(pose=start[name], speed=args.approach_speed)
        time.sleep(1.0)

        if not args.no_grip:
            print('\nOpening both grippers -- place the cable between them now.')
            grip(1.0)
            input('  press ENTER when the cable is in place (Ctrl-C to abort)... ')
            print('Closing both grippers...')
            grip(0.0)
            time.sleep(1.5)

        rate = robots['left'].node.create_rate(args.rate)
        print(f'\nRunning freestyle motion for {args.duration:.0f} s '
              f'at {args.rate:.0f} Hz. Ctrl-C to stop early.\n')
        t, t0, i = 0.0, time.time(), 0
        while t < args.duration:
            poses, sep_cmd = planner.pose_step(t)
            for name, robot in robots.items():
                robot.set_target(pose=poses[name])

            ml = robots['left'].end_effector_pose.position
            mr = robots['right'].end_effector_pose.position
            sep_meas = float(np.linalg.norm(planner.to_world(mr) - ml))
            worst_sep = max(worst_sep, sep_meas)

            log['t'].append(t)
            log['wall'].append(time.time())
            log['q_left'].append(robots['left'].joint_values[:7].copy())
            log['q_right'].append(robots['right'].joint_values[:7].copy())
            log['tcp_meas_left'].append(ml.copy())
            log['tcp_meas_right'].append(mr.copy())
            log['tcp_cmd_left'].append(poses['left'].position.copy())
            log['tcp_cmd_right'].append(poses['right'].position.copy())
            log['sep_cmd'].append(sep_cmd)
            log['sep_meas'].append(sep_meas)

            if i % int(args.rate) == 0:
                flag = '  <-- OVER' if sep_meas > args.max_sep + 0.02 else ''
                print(f'  t={t:6.1f}s  sep cmd {sep_cmd*100:5.1f} cm  '
                      f'meas {sep_meas*100:5.1f} cm{flag}')
            i += 1
            rate.sleep()
            t += dt
        print(f'\nMotion complete ({time.time() - t0:.1f} s wall).')

    except KeyboardInterrupt:
        print('\nCtrl-C -- stopping.')
    finally:
        if worst_sep > 0.0:
            print(f'Worst measured separation: {worst_sep*100:.1f} cm '
                  f'(limit {args.max_sep*100:.0f} cm)')
        _save_log(args, log, planner)
        try:
            if not args.no_grip:
                print('Opening grippers to release the cable...')
                grip(1.0)
            if not args.no_home:
                print('Returning both arms home...')
                for robot in robots.values():
                    robot.home()
        except Exception as exc:
            print(f'  shutdown motion failed: {exc}')
        for robot in robots.values():
            try:
                robot.shutdown()
            except Exception:
                pass
    return 0


def _save_log(args, log, planner: DualArmPlanner):
    if not log['t']:
        print('No samples recorded; nothing to save.')
        return
    out = Path(args.out) / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in log.items()}
    arrays['T_right_base2world'] = planner.T_rb2w
    arrays['max_sep_m'] = np.float64(planner.prm.sep_max)
    np.savez(out / 'motion_log.npz', **arrays)
    sep = arrays['sep_meas']
    print(f'Saved {len(sep)} ticks to {out / "motion_log.npz"}')
    print(f'  measured separation {sep.min()*100:.1f} .. {sep.max()*100:.1f} cm '
          f'(limit {planner.prm.sep_max*100:.0f} cm)')


# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Freestyle dual-arm test motion for the live DLO tracker.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dry_run', action='store_true',
                   help='simulate the trajectory and report limits; no ROS, no robots')
    p.add_argument('--calib', default=str(DEFAULT_CALIB),
                   help='transform_ee_cam_world.npz -- relates the two robot bases')
    p.add_argument('--impedance', default=str(DEFAULT_IMPEDANCE))
    p.add_argument('--out', default=str(DEFAULT_OUT))
    p.add_argument('--duration', type=float, default=180.0, help='seconds of motion')
    p.add_argument('--rate', type=float, default=50.0, help='control Hz')
    p.add_argument('--max_sep', type=float, default=0.55,
                   help='hard ceiling on gripper separation, m')
    p.add_argument('--sep_nom', type=float, default=0.46,
                   help='mean gripper separation, m')
    p.add_argument('--sep_amp', type=float, default=0.07,
                   help='how much the separation breathes, m')
    p.add_argument('--mid_z', type=float, default=0.36,
                   help='height of the gripper midpoint above the robot bases, m')
    p.add_argument('--speed', type=float, default=1.0,
                   help='multiplies every trajectory frequency')
    p.add_argument('--max_speed', type=float, default=0.15,
                   help='ceiling on commanded TCP speed, m/s. Lower this if the '
                        'tracker shows arm streaks along the direction of motion '
                        '(README_ZED.md §7 -- that is joint lag, not calibration)')
    p.add_argument('--tilt_deg', type=float, default=8.0,
                   help='wrist wobble amplitude, deg; 0 pins the gripper straight down')
    p.add_argument('--yaw_left', type=float, default=0.0,
                   help='rotate the left gripper about vertical, deg')
    p.add_argument('--yaw_right', type=float, default=0.0)
    p.add_argument('--approach_speed', type=float, default=0.08,
                   help='m/s for the initial move to the start pose')
    p.add_argument('--no_grip', action='store_true',
                   help='skip the open / wait-for-cable / close sequence')
    p.add_argument('--no_home', action='store_true',
                   help='skip homing at start and end')
    p.add_argument('--timeout', type=float, default=15.0)
    args = p.parse_args()

    T_rb2w = load_world_transform(args.calib)
    t, R = T_rb2w[:3, 3], T_rb2w[:3, :3]
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    print(f'Calibration: {args.calib}')
    print(f'  right base in left-base frame: {np.round(t, 4)} m, '
          f'yaw {yaw:+.1f} deg  -> bases {np.linalg.norm(t):.4f} m apart')

    prm = make_params(T_rb2w, args)
    planner = DualArmPlanner(T_rb2w, prm, dt=1.0 / args.rate,
                             yaw_left=np.deg2rad(args.yaw_left),
                             yaw_right=np.deg2rad(args.yaw_right))
    print(f'  gripper midpoint parked at {np.round(prm.mid0, 4)} m (world = left base)')

    if args.dry_run:
        return dry_run(planner, args.duration, 1.0 / args.rate)
    return run_live(args, planner)


if __name__ == '__main__':
    sys.exit(main())
