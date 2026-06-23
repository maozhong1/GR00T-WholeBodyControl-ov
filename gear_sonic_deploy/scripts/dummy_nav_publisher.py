#!/usr/bin/env python3
"""
Dummy navigation publisher for the ROS2 input path of g1_deploy_onnx_ref.

Publishes msgpack-serialised ControlGoalMsg payloads on the topic
`ControlPolicy/upper_body_pose` (std_msgs/ByteMultiArray), exercising the
`navigate_cmd` + `base_height_command` plumbing parsed by
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/ros2_input_handler.hpp`.

Use case: simulate an upstream autonomous-navigation node sending cmd_vel-style
velocity commands to drive the locomotion planner, with `base_height_command`
pinned at 0.78 (standing).

Usage:
    # Terminal A — start deploy in ROS2 mode (planner required)
    cd gear_sonic_deploy
    source scripts/setup_env.sh
    bash deploy.sh sim --input-type ros2 \
        --planner planner/target_vel/V2/planner_sonic.onnx

    # Terminal B — source ROS2 + run this script
    source /opt/ros/jazzy/setup.bash
    pip install msgpack msgpack-numpy
    python3 scripts/dummy_nav_publisher.py

CLI:
    --rate HZ          Publish rate (default 10)
    --vx M/S           Forward linear velocity for the "drive" phase (default 0.3)
    --vy M/S           Lateral linear velocity for the "drive" phase (default 0.0)
    --wz RAD/S         Yaw angular velocity for the "turn" phase (default 0.3)
    --base-height M    base_height_command pinned value (default 0.78)
    --locomotion-mode  0 = slow walk, 1 = fast walk (default 1)
    --pattern NAME     square | forward | turn | manual (default square)
    --segment SEC      Phase duration for `square` / `forward` / `turn` patterns (default 5.0)
    --auto-start       Pulse `toggle_policy_action` at t=2s to auto-trigger control
    --toggle-initial   Initial-pulse delay in seconds (default 2.0; only used if --auto-start)
    --toggle-interval  Re-pulse `toggle_policy_action` every N seconds (default 0 = disabled)
    --max-toggles      Stop pulsing after N pulses (default 0 = unlimited)
    --verbose          Log every published message instead of every 10

NOTE on toggle semantics: `toggle_policy_action` is edge-triggered AND
*toggles* between START and STOP on each pulse. The 1st pulse starts
control; the 2nd stops it; the 3rd restarts; …  By default we send a
single pulse only. Pass `--toggle-interval N` (>0) to re-fire every N
seconds — useful for a deploy that starts AFTER this publisher — but
remember to disable it once deploy logs 'START control', otherwise the
next pulse will STOP control again.
"""
import argparse
import threading
import time

import msgpack
import msgpack_numpy as mnp
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import ByteMultiArray

TOPIC = "ControlPolicy/upper_body_pose"


def pack_payload(*, navigate_cmd, base_height, locomotion_mode, toggle):
    """Build a msgpack-encoded ControlGoalMsg matching the C++ parser."""
    payload = {
        "navigate_cmd": list(navigate_cmd),                 # [vx, vy, wz]
        "base_height_command": float(base_height),
        "toggle_policy_action": bool(toggle),
        "locomotion_mode": int(locomotion_mode),
        "ros_timestamp": time.time(),
        "valid": True,
        # Identity wrist / head poses — the planner doesn't consume them in this
        # use case but parser is happy to skip missing keys.
    }
    return msgpack.packb(payload, default=mnp.encode)


def to_byte_multi_array(packed_bytes):
    """ByteMultiArray.data is bytes[] (each element 1 byte) — match
    decoupled_wbc.control.utils.ros_utils.ROSMsgPublisher serialisation."""
    msg = ByteMultiArray()
    msg.data = [bytes([b]) for b in packed_bytes]
    return msg


def pattern_square(t, seg, vx, vy, wz):
    """4 phases of `seg` seconds each: forward / turn-left / forward / turn-left."""
    phase = int(t // seg) % 4
    if phase == 0:
        return [vx, 0.0, 0.0]
    if phase == 1:
        return [0.0, 0.0, wz]
    if phase == 2:
        return [vx, 0.0, 0.0]
    return [0.0, 0.0, wz]


def pattern_forward(t, seg, vx, vy, wz):
    """Drive forward for `seg` s, stop for `seg` s, repeat."""
    phase = int(t // seg) % 2
    return [vx, 0.0, 0.0] if phase == 0 else [0.0, 0.0, 0.0]


def pattern_turn(t, seg, vx, vy, wz):
    """Spin in place: +wz for `seg` s, then -wz."""
    phase = int(t // seg) % 2
    return [0.0, 0.0, wz if phase == 0 else -wz]


def pattern_manual(t, seg, vx, vy, wz):
    """Always emit the same constant command (vx, vy, wz)."""
    return [vx, vy, wz]


PATTERNS = {
    "square": pattern_square,
    "forward": pattern_forward,
    "turn": pattern_turn,
    "manual": pattern_manual,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rate", type=float, default=10.0, help="Publish rate in Hz")
    p.add_argument("--vx", type=float, default=0.3, help="Forward linear velocity (m/s)")
    p.add_argument("--vy", type=float, default=0.0, help="Lateral linear velocity (m/s)")
    p.add_argument("--wz", type=float, default=0.3, help="Yaw angular velocity (rad/s)")
    p.add_argument("--base-height", type=float, default=0.78, help="base_height_command (m)")
    p.add_argument("--locomotion-mode", type=int, choices=[0, 1], default=1,
                   help="0 = slow walk (per-direction speed), 1 = fast walk (planner default speed)")
    p.add_argument("--pattern", choices=list(PATTERNS.keys()), default="square",
                   help="Command pattern over time")
    p.add_argument("--segment", type=float, default=5.0, help="Phase duration (s) for time-varying patterns")
    p.add_argument("--auto-start", action="store_true",
                   help="Pulse toggle_policy_action=True (replaces manual operator 'start')")
    p.add_argument("--toggle-initial", type=float, default=2.0,
                   help="Delay before first toggle pulse (s). Only used when --auto-start is set.")
    p.add_argument("--toggle-interval", type=float, default=0.0,
                   help="Re-pulse every N seconds (default 0 = single pulse only). "
                        "WARNING: each pulse FLIPS start<->stop, so a late deploy catching a 2nd pulse "
                        "stops control. Use a positive value only when deploy may start AFTER this script.")
    p.add_argument("--max-toggles", type=int, default=0,
                   help="Stop pulsing after N pulses (0 = unlimited).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    rclpy.init()
    node = rclpy.create_node("dummy_nav_publisher")

    # Reliable QoS, depth=1 — matches the C++ subscriber side (default rclcpp QoS).
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.VOLATILE,
    )
    pub = node.create_publisher(ByteMultiArray, TOPIC, qos)

    # Spin in a background thread so node logging works.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    pattern_fn = PATTERNS[args.pattern]
    period = 1.0 / args.rate
    t0 = time.monotonic()
    msg_count = 0
    next_toggle_t = args.toggle_initial if args.auto_start else None
    toggle_count = 0

    node.get_logger().info(
        f"Publishing on '{TOPIC}' @ {args.rate:g} Hz | pattern={args.pattern} "
        f"vx={args.vx} vy={args.vy} wz={args.wz} "
        f"base_height={args.base_height} locomotion_mode={args.locomotion_mode}"
    )
    if args.auto_start:
        node.get_logger().info(
            f"auto-start: first toggle at t={args.toggle_initial:.1f}s; "
            f"re-pulse every {args.toggle_interval:.1f}s "
            f"(0 = once); max_toggles={args.max_toggles or 'unlimited'}"
        )
        node.get_logger().warning(
            "toggle_policy_action FLIPS start<->stop on each pulse. "
            "Once deploy logs 'START control', restart this script with "
            "--toggle-interval 0 (or use --max-toggles 1) to avoid an unintended STOP."
        )

    try:
        while rclpy.ok():
            t = time.monotonic() - t0
            navigate_cmd = pattern_fn(t, args.segment, args.vx, args.vy, args.wz)

            # Edge-triggered start pulse: fire at next_toggle_t, then re-arm by toggle_interval.
            toggle = False
            if next_toggle_t is not None and t >= next_toggle_t:
                toggle = True
                toggle_count += 1
                node.get_logger().info(
                    f"Sending toggle_policy_action=True (pulse #{toggle_count} at t={t:.2f}s)"
                )
                if args.max_toggles and toggle_count >= args.max_toggles:
                    next_toggle_t = None
                    node.get_logger().info("Reached --max-toggles; no more pulses will be sent.")
                elif args.toggle_interval > 0:
                    next_toggle_t = t + args.toggle_interval
                else:
                    next_toggle_t = None  # one-shot

            packed = pack_payload(
                navigate_cmd=navigate_cmd,
                base_height=args.base_height,
                locomotion_mode=args.locomotion_mode,
                toggle=toggle,
            )
            pub.publish(to_byte_multi_array(packed))
            msg_count += 1

            log_now = args.verbose or (msg_count % max(1, int(args.rate)) == 0)
            if log_now:
                node.get_logger().info(
                    f"[{t:6.2f}s] navigate_cmd=[{navigate_cmd[0]:+.2f}, "
                    f"{navigate_cmd[1]:+.2f}, {navigate_cmd[2]:+.2f}] "
                    f"base_height={args.base_height:.2f} mode={args.locomotion_mode}"
                )

            time.sleep(period)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
