"""Open both Franka grippers, send both Franka arms home, then close both grippers.

This script initializes both robots, opens both grippers first, waits briefly,
commands both arms to their home position, and finally closes both grippers.
"""

from __future__ import annotations

import time

from std_msgs.msg import Float64MultiArray
from crisp_py.robot import Robot


def publish_gripper(robot: Robot, side: str, value: float, repeat: int = 8) -> None:
    """
    Publish gripper position command several times.

    value = 1.0 means open
    value = 0.0 means close
    """
    topic = f"/{side}/gripper/gripper_position_controller/commands"

    pub = robot.node.create_publisher(
        Float64MultiArray,
        topic,
        10,
    )

    msg = Float64MultiArray()
    msg.data = [float(value)]

    # Publish several times because the first message may be lost
    # before the publisher/subscriber connection is fully ready.
    for _ in range(repeat):
        pub.publish(msg)
        time.sleep(0.1)


def gripper_open(robot: Robot, side: str) -> None:
    print(f"Opening {side} gripper...")
    publish_gripper(robot, side, 1.0)
    time.sleep(1.5)


def gripper_close(robot: Robot, side: str) -> None:
    print(f"Closing {side} gripper...")
    publish_gripper(robot, side, 0.0)
    time.sleep(1.5)


def main() -> None:
    # Initialize robots
    left_arm = Robot(namespace="/left")
    right_arm = Robot(namespace="/right")

    left_arm.wait_until_ready()
    right_arm.wait_until_ready()

    try:
        print("Opening both grippers first...")
        gripper_open(left_arm, "left")
        gripper_open(right_arm, "right")
        print("Both grippers should now be open.")

        print("Sending both arms to home position...")
        left_arm.home()
        right_arm.home()
        print("Both arms are now at home.")

        # Final step: close both grippers
        print("Closing both grippers at the end...")
        gripper_close(left_arm, "left")
        gripper_close(right_arm, "right")
        print("Both grippers should now be closed.")

    finally:
        left_arm.shutdown()
        right_arm.shutdown()


if __name__ == "__main__":
    main()