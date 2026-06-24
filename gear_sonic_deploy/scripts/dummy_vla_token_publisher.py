#!/usr/bin/env python3
"""
Dummy VLA token publisher for the ROS2 input path of g1_deploy_onnx_ref.

Publishes a ZMQ-Protocol-v4-compatible packed payload on the topic
`ControlPolicy/vla_token` (std_msgs/ByteMultiArray). The wire format is:

    [1280-byte JSON header (null-padded)] [concatenated binary fields]

Header (JSON):
    { "v": 4, "endian": "le", "count": 1,
      "fields": [
        { "name": "token_state",      "dtype": "f32",  "shape": [N] },
        { "name": "toggle_vla_mode",  "dtype": "bool", "shape": [1] },
        { "name": "frame_index",      "dtype": "i64",  "shape": [1] }
      ] }

This is intentionally identical to ZMQEndpointInterface's Protocol v4 layout,
so the same VLA producer can switch transports without re-encoding the
payload.  Parsed on the C++ side by
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/ros2_input_handler.hpp::parse_packed_vla_token().

Use case: simulate the VLA inference process pushing latent action tokens
to drive the decoder, plus the toggle_vla_mode pulse that switches the
deploy between encoder+planner (navigate) and decoder+token (manipulation).

Usage:
    # Terminal A — start deploy in ROS2 mode (encoder + planner default)
    cd gear_sonic_deploy
    source scripts/setup_env.sh
    bash deploy.sh sim --input-type ros2 \
        --planner planner/target_vel/V2/planner_sonic.onnx

    # Terminal B — drive navigation (Step 1 of the verification flow)
    source /opt/ros/jazzy/setup.bash
    python3 scripts/dummy_nav_publisher.py --auto-start

    # Terminal C — push VLA tokens (Step 2/3)
    python3 scripts/dummy_vla_token_publisher.py
        # → at t=2s sends toggle_vla_mode=True (ENTER VLA mode)
        # → on Ctrl+C with --toggle-on-exit, sends a 2nd pulse (EXIT)

CLI:
    --rate HZ              Publish rate (default 50, matches run_vla_inference.py action_publish_rate)
    --token-dim N          Token vector dimension (default 64, matches encoder.dimension)
    --token-mode MODE      zeros | random | sine (default sine)
    --toggle-initial SEC   Send toggle_vla_mode=True at this time (default 2.0; -1 = disabled)
    --toggle-on-exit       On Ctrl+C, send a 2nd toggle pulse before shutting down
    --with-hand-joints     Also include left/right_hand_joints (7-DOF Dex3 each, f32 [7])
    --hand-mode MODE       zeros | sine (default zeros) — only used when --with-hand-joints
    --verbose              Log every published message instead of every `rate` msgs
"""
import argparse
import json
import struct
import threading
import time

import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import ByteMultiArray

TOPIC = "ControlPolicy/vla_token"
HEADER_SIZE = 1280
PROTOCOL_VERSION = 4


def build_packed_payload(
    token: np.ndarray,
    toggle_vla_mode: bool,
    frame_index: int,
    left_hand: np.ndarray | None = None,
    right_hand: np.ndarray | None = None,
) -> bytes:
    """Construct a ZMQ-v4-compatible packed payload (header + binary fields).

    Hand-joint fields are optional; when provided they must each be a length-7
    float32 vector. The C++ side accepts shape [7] or [N,7]; we always send [7].
    """
    token_f32 = token.astype("<f4", copy=False)
    fields = [
        {"name": "token_state", "dtype": "f32", "shape": [int(token_f32.size)]},
        {"name": "toggle_vla_mode", "dtype": "bool", "shape": [1]},
        {"name": "frame_index", "dtype": "i64", "shape": [1]},
    ]
    if left_hand is not None:
        fields.append({"name": "left_hand_joints", "dtype": "f32", "shape": [7]})
    if right_hand is not None:
        fields.append({"name": "right_hand_joints", "dtype": "f32", "shape": [7]})

    header = {"v": PROTOCOL_VERSION, "endian": "le", "count": 1, "fields": fields}
    header_bytes = json.dumps(header).encode("utf-8")
    if len(header_bytes) > HEADER_SIZE:
        raise RuntimeError(f"header too large: {len(header_bytes)} > {HEADER_SIZE}")
    header_bytes = header_bytes + b"\x00" * (HEADER_SIZE - len(header_bytes))

    body = bytearray()
    body += token_f32.tobytes()
    body += struct.pack("<B", 1 if toggle_vla_mode else 0)  # bool[1]
    body += struct.pack("<q", int(frame_index))  # i64[1]
    if left_hand is not None:
        lh = np.asarray(left_hand, dtype="<f4").reshape(-1)
        if lh.size != 7:
            raise ValueError(f"left_hand must be length 7, got {lh.size}")
        body += lh.tobytes()
    if right_hand is not None:
        rh = np.asarray(right_hand, dtype="<f4").reshape(-1)
        if rh.size != 7:
            raise ValueError(f"right_hand must be length 7, got {rh.size}")
        body += rh.tobytes()
    return bytes(header_bytes + body)


def to_byte_multi_array(packed: bytes) -> ByteMultiArray:
    """ByteMultiArray.data is bytes[] (each element 1 byte) — match
    decoupled_wbc.control.utils.ros_utils.ROSMsgPublisher serialisation."""
    msg = ByteMultiArray()
    msg.data = [bytes([b]) for b in packed]
    return msg


def make_token(mode: str, dim: int, t: float, rng: np.random.Generator) -> np.ndarray:
    if mode == "zeros":
        return np.zeros(dim, dtype=np.float32)
    if mode == "random":
        return rng.normal(0.0, 0.1, size=dim).astype(np.float32)
    # sine: smooth, replayable token sequence (each dim has its own phase)
    phases = np.arange(dim, dtype=np.float32) * 0.1
    return (0.1 * np.sin(2 * np.pi * 0.2 * t + phases)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Publish rate in Hz (default 50, matches run_vla_inference.py action_publish_rate)",
    )
    p.add_argument("--token-dim", type=int, default=64, help="Token vector dimension")
    p.add_argument(
        "--token-mode",
        choices=["zeros", "random", "sine"],
        default="sine",
        help="How to synthesise the token stream",
    )
    p.add_argument(
        "--toggle-initial",
        type=float,
        default=2.0,
        help="Send toggle_vla_mode=True at t=N seconds (one-shot). -1 = never send.",
    )
    p.add_argument(
        "--toggle-on-exit",
        action="store_true",
        help="On Ctrl+C, send a 2nd toggle_vla_mode=True before shutdown (EXIT VLA mode cleanly)",
    )
    p.add_argument(
        "--with-hand-joints",
        action="store_true",
        help="Also publish left_hand_joints and right_hand_joints (7 DOF each, f32 [7])",
    )
    p.add_argument(
        "--hand-mode",
        choices=["zeros", "sine"],
        default="zeros",
        help="How to synthesise hand-joint targets (only used with --with-hand-joints)",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    rclpy.init()
    node = rclpy.create_node("dummy_vla_token_publisher")
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.VOLATILE,
    )
    pub = node.create_publisher(ByteMultiArray, TOPIC, qos)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    rng = np.random.default_rng(0)
    period = 1.0 / args.rate
    t0 = time.monotonic()
    next_toggle = args.toggle_initial if args.toggle_initial >= 0 else None
    msg_count = 0

    def make_hand(t: float) -> np.ndarray | None:
        if not args.with_hand_joints:
            return None
        if args.hand_mode == "sine":
            phases = np.arange(7, dtype=np.float32) * 0.3
            return (0.2 * np.sin(2 * np.pi * 0.5 * t + phases)).astype(np.float32)
        return np.zeros(7, dtype=np.float32)

    node.get_logger().info(
        f"Publishing on '{TOPIC}' @ {args.rate:g} Hz | dim={args.token_dim} "
        f"mode={args.token_mode} hand_joints={'on' if args.with_hand_joints else 'off'}"
    )
    if next_toggle is not None:
        node.get_logger().info(
            f"Will send toggle_vla_mode=True (ENTER VLA) at t={next_toggle:.1f}s"
        )
    if args.toggle_on_exit:
        node.get_logger().info("Will send toggle_vla_mode=True (EXIT VLA) on Ctrl+C")

    try:
        while rclpy.ok():
            t = time.monotonic() - t0
            token = make_token(args.token_mode, args.token_dim, t, rng)
            left_hand = make_hand(t)
            right_hand = make_hand(t)

            toggle = False
            if next_toggle is not None and t >= next_toggle:
                toggle = True
                next_toggle = None  # one-shot
                node.get_logger().info(
                    f"Sending toggle_vla_mode=True (ENTER VLA) at t={t:.2f}s"
                )

            packed = build_packed_payload(
                token, toggle, msg_count, left_hand=left_hand, right_hand=right_hand
            )
            pub.publish(to_byte_multi_array(packed))
            msg_count += 1

            log_now = args.verbose or (msg_count % max(1, int(args.rate)) == 0)
            if log_now:
                hand_msg = (
                    f" hand_l[0]={left_hand[0]:+.3f}" if left_hand is not None else ""
                )
                node.get_logger().info(
                    f"[{t:6.2f}s] token[0]={token[0]:+.3f} frame={msg_count} "
                    f"toggle={'ENTER' if toggle else '-'}{hand_msg}"
                )

            time.sleep(period)
    except KeyboardInterrupt:
        if args.toggle_on_exit:
            node.get_logger().info("Sending toggle_vla_mode=True (EXIT VLA) before shutdown")
            zero_token = np.zeros(args.token_dim, dtype=np.float32)
            zero_hand = np.zeros(7, dtype=np.float32) if args.with_hand_joints else None
            pub.publish(
                to_byte_multi_array(
                    build_packed_payload(
                        zero_token, True, msg_count,
                        left_hand=zero_hand, right_hand=zero_hand,
                    )
                )
            )
            time.sleep(0.2)
        node.get_logger().info("Interrupted, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
