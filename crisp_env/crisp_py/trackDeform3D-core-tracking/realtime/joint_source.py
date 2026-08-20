"""Live joint-configuration sources for the rendered-arm reference.

Same pattern as frame_source.py: one small interface, two implementations.

    CrispJointSource    -- the real robots, through crisp_py. Each arm is one
                           crisp_py Robot subscribed to its JointState topic
                           (robot.py:122-130), so latest() is just an array
                           copy of the newest received joints -- no waiting.
    ConstantJointSource -- fixed joint vectors, for replay rehearsal and
                           tests (no ROS, no robots).

crisp_py / rclpy are imported lazily inside CrispJointSource.start(), so the
replay and test paths run in the base trackdeform3d environment.

The snapshot is the LATEST received state, not the state at the camera's
exposure time: at 30 fps the joints lag the pixels by up to one frame plus
the camera pipeline latency. armdiff_segmenter.py absorbs that with its
arm-mask dilation (arm_dilate_px); slow the arms or raise the dilation if
residual arm streaks appear along the direction of motion.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from realtime.arm_reference import FINGER_OPEN


@dataclass
class JointSnapshot:
    """Joint state of every arm at (approximately) one instant."""
    q: Dict[str, np.ndarray]                     # {'left': (7,) rad, ...}
    finger_q: Dict[str, float] = field(default_factory=dict)   # per-finger m


class JointSource:
    """start() -> latest() per frame -> stop()."""

    def start(self) -> 'JointSource':
        return self

    def stop(self):
        pass

    def latest(self) -> JointSnapshot:
        raise NotImplementedError

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class ConstantJointSource(JointSource):
    """Fixed configuration (default: the fr3 'ready' pose for both arms).
    The rehearsal path: exercises FK + rendering + subtraction end to end
    on replay data, where no live robots exist."""

    READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])

    def __init__(self, q_by_arm: Optional[Dict[str, np.ndarray]] = None,
                 finger_q: float = FINGER_OPEN):
        if q_by_arm is None:
            q_by_arm = {'left': self.READY.copy(), 'right': self.READY.copy()}
        self._snap = JointSnapshot(
            q={k: np.asarray(v, dtype=np.float64) for k, v in q_by_arm.items()},
            finger_q={k: float(finger_q) for k in q_by_arm})

    def latest(self) -> JointSnapshot:
        return self._snap


class CrispJointSource(JointSource):
    """Both arms' joints from the crisp_py JointState subscriptions.

    namespaces: the ROS namespaces of the arms, e.g. ('left', 'right') --
    the same names dual_green_pick.py uses (Robot(namespace='/left')).
    The dict keys of the snapshot are the namespace names with slashes
    stripped, matching the arm names the ArmDepthRenderer was built with.

    finger_q: rendered per-finger travel in meters. The default (fully open)
    draws the LARGEST finger silhouette, which over-subtracts at most a few
    extra pixels next to the gripper; pass a Gripper-fed callable via
    finger_q_fn for exact widths if that matters.

    publish_targets: leave FALSE. crisp_py's Robot is not a passive observer --
    its _callback_current_pose latches `_target_pose = current_pose` on the
    first pose it receives (robot.py:415-417) and a 50 Hz timer then republishes
    that frozen pose to <ns>/target_pose forever (robot.py:145-152). A joint
    SOURCE that does this is commanding the arm to hold still: any other node
    driving the same topic (a scripted test motion, a teleop, a policy) ends up
    interleaved with a stale hold-pose at 50 Hz and the arm judders between the
    two. So start() cancels this node's timers, which costs nothing here -- the
    joint state arrives on a subscription, driven by the executor, and needs no
    timer at all. The only casualty is CallbackMonitor's staleness warning.
    start() also DESTROYS the target publishers themselves: the crisp
    controllers' guard counts publisher endpoints via DDS discovery ("Topic
    'target_pose' has 2 publishers (expected max: 1)"), so even a publisher
    that never sends makes the controller ignore all commands -- including the
    real motion script's.
    """

    def __init__(self, namespaces=('left', 'right'), finger_q: float = FINGER_OPEN,
                 finger_q_fn=None, timeout: float = 10.0,
                 publish_targets: bool = False):
        self.namespaces = tuple(namespaces)
        self.finger_q = float(finger_q)
        self.finger_q_fn = finger_q_fn           # optional: name -> finger m
        self.timeout = timeout
        self.publish_targets = bool(publish_targets)
        self._robots = {}

    def start(self) -> 'CrispJointSource':
        from crisp_py.robot import Robot         # lazy: needs rclpy + crisp_py

        for ns in self.namespaces:
            name = ns.strip('/')
            # namespace=name, NOT f'/{name}'. Both forms resolve the same topics,
            # but Robot builds its joint-name prefix as f'{namespace}_'
            # (robot.py:72), so a leading slash gives '/left_' while the driver
            # publishes 'left_fr3_joint1'. _callback_current_joint then filters
            # on removeprefix(self._prefix), matches nothing, and leaves
            # _current_joint at its np.zeros() initialisation (robot.py:439-450).
            # joint_values would be all zeros -- no exception, no warning, and
            # is_ready() still passes -- so the renderer would draw both arms
            # bolt upright in the q=0 pose and subtract the wrong pixels forever.
            robot = Robot(namespace=name, name=f'armdiff_{name}_joints')
            if not self.publish_targets:
                # Before wait_until_ready(), not after: the publish callback is a
                # no-op while _target_pose is None, so cancelling now means this
                # node never emits a single target message. Cancelling afterwards
                # would still leak the handful published while we waited.
                for timer in list(robot.node.timers):
                    timer.cancel()
                # Cancelling the timers silences this node, but the crisp
                # controllers' single-command-source guard counts DDS publisher
                # ENDPOINTS, not messages: a second (mute) publisher on
                # <ns>/target_pose makes the controller ignore EVERY command,
                # including a real motion script's. So the endpoints must go.
                for pub in (robot._target_pose_publisher,
                            robot._target_joint_publisher,
                            robot._target_wrench_publisher):
                    robot.node.destroy_publisher(pub)
            self._robots[name] = robot
        for name, robot in self._robots.items():
            robot.wait_until_ready(timeout=self.timeout)
        return self

    def latest(self) -> JointSnapshot:
        if not self._robots:
            raise RuntimeError('CrispJointSource.start() was not called')
        q = {name: robot.joint_values[:7]
             for name, robot in self._robots.items()}
        fq = {name: (self.finger_q_fn(name) if self.finger_q_fn else self.finger_q)
              for name in self._robots}
        return JointSnapshot(q=q, finger_q=fq)

    def stop(self):
        for robot in self._robots.values():
            try:
                robot.shutdown()
            except Exception:
                pass
        self._robots = {}
