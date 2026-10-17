#!/usr/bin/env python3
"""Client library for the controller's ZeroMQ interface.

Implements the binary frame protocol from PROTOCOL.md: the wire
constants, the frame codec, and the Shell client (latest-only telemetry
subscriber plus a command request socket with state waiters). Also owns
the endpoint command-line flags shared by every tool in this directory
(add_endpoint_arguments / resolve_endpoints).
"""

import struct
import time

import zmq

MAGIC = 0x52
VERSION = 0x03
TYPE_TELEMETRY = 0x01
TYPE_EVENT = 0x10
TYPE_SET_TARGET = 0x11
TYPE_SET_JOINT_VELOCITIES = 0x12
TYPE_SET_TWIST = 0x13
TYPE_SET_MODE = 0x14
TYPE_MOVE_TO_SAFE = 0x15
TYPE_ZERO_MOTOR = 0x16
TYPE_ZERO_ALL = 0x17
TYPE_GRIPPER_ACTIVATE = 0x18
TYPE_GRIPPER_COMMAND = 0x19
TYPE_GRIPPER_STOP = 0x1A
TYPE_SET_JOINT_VELOCITIES_RAMPED = 0x1B
TYPE_SET_EE_TWIST_RAMPED = 0x1C
TYPE_GET_MODEL = 0x1D
TYPE_GET_ASSET = 0x1E
TYPE_SET_JOINT_POSITIONS = 0x1F
TYPE_REPLY = 0x20
TYPE_MODEL = 0x21
TYPE_ASSET = 0x22
TYPE_GET_TARGET_FRAME = 0x23
TYPE_TARGET_FRAME = 0x24
TYPE_SET_TARGET_FRAME = 0x25
# Only URDF links named like this can be the controlled tip.
TARGET_FRAME_PREFIX = "target_frame"
# Planner frames: answered by the planner binary; a bare controller
# replies UNSUPPORTED ("planner frame: no planner in this server").
TYPE_SET_JOINT_TARGET = 0x30
TYPE_SET_POSE_TARGET = 0x31
TYPE_CANCEL_TARGET = 0x32
TYPE_GET_PLANNER_STATUS = 0x33
TYPE_PLANNER_STATUS = 0x34
TYPE_SET_JOINT_TRAJECTORY = 0x35
MAX_TRAJECTORY_WAYPOINTS = 4096

EVENTS = {
    "power_on": 1,
    "run": 2,
    "hold": 3,
    "power_off": 4,
    "clear": 5,
    "estop": 6,
    "release_estop": 7,
}

CONTROLLER_STATES = {0: "power_off", 1: "power_on", 2: "running", 3: "error"}
PLANNER_STATES = {0: "idle", 1: "preparing", 2: "executing", 3: "planning"}
CONTROL_MODES = {0: "pose", 1: "joint_vel", 2: "world_twist", 3: "ee_twist",
                 4: "jv_ramp", 5: "eev_ramp", 6: "joint_pos"}
# "twist" is kept as an alias for world_twist (the pre-rename mode name).
MODE_CODES = {"pose": 0, "jv": 1, "joint_vel": 1,
              "twist": 2, "world_twist": 2, "ee_twist": 3,
              "jvr": 4, "jv_ramp": 4, "eevr": 5, "eev_ramp": 5,
              "jp": 6, "joint_position": 6}
SESSION_STATES = {0: "Disconnected", 1: "Idle", 2: "Releasing", 3: "Held",
                  4: "Arming", 5: "Armed", 6: "Active", 7: "Fault", 8: "EstopLatched"}
FIRMWARE_STATES = {0: "INACTIVE", 1: "ACTIVE", 2: "HOLD", 3: "ESTOP"}
REPLY_STATUS = {0: "OK", 1: "MALFORMED", 2: "UNSUPPORTED", 3: "REJECTED_STATE", 4: "FAILED"}

DEFAULT_HOST = "127.0.0.1"
PUB_PORT = 5555
CMD_PORT = 5556

# Header + fixed telemetry payload (328 bytes total, PROTOCOL.md v3).
TELEMETRY_FIXED = struct.Struct("<3B Q 7B 18d 6h 14d I 5f 4B 3f H")


def add_endpoint_arguments(parser):
    """Add the connection flags shared by the tools in this directory."""
    parser.add_argument(
        "--host", metavar="HOST",
        help=f"controller host: shorthand for --pub tcp://HOST:{PUB_PORT} "
             f"--cmd tcp://HOST:{CMD_PORT} (default {DEFAULT_HOST})")
    parser.add_argument("--pub", metavar="EP",
                        help="telemetry PUB endpoint; overrides --host")
    parser.add_argument("--cmd", metavar="EP",
                        help="command REP endpoint; overrides --host")


def resolve_endpoints(args):
    """Endpoints from --pub/--cmd, the --host shorthand, or local defaults."""
    host = args.host or DEFAULT_HOST
    pub = args.pub or f"tcp://{host}:{PUB_PORT}"
    cmd = args.cmd or f"tcp://{host}:{CMD_PORT}"
    return pub, cmd


def parse_telemetry(data: bytes):
    if len(data) < TELEMETRY_FIXED.size:
        return None
    fields = TELEMETRY_FIXED.unpack_from(data)
    magic, version, ftype = fields[0], fields[1], fields[2]
    if magic != MAGIC or version != VERSION or ftype != TYPE_TELEMETRY:
        return None
    fault_len = fields[-1]
    fault = data[TELEMETRY_FIXED.size:TELEMETRY_FIXED.size + fault_len].decode(
        "utf-8", "replace")
    doubles = fields[11:29]
    return {
        "sequence": fields[3],
        "controller_state": fields[4],
        "control_mode": fields[5],
        "settling": fields[6],
        "session_state": fields[7],
        "firmware_state": fields[8],
        "firmware_reason": fields[9],
        "motion_active": fields[10],
        "q": doubles[0:6],
        "qdot_measured": doubles[6:12],
        "qdot_commanded": doubles[12:18],
        "current_mA": fields[29:35],
        "tcp_pose": fields[35:42],
        "target_pose": fields[42:49],
        "firmware_cycle": fields[49],
        "feedback_age_ms": fields[50],
        "input_age_ms": fields[51],
        "loop_last_ms": fields[52],
        "loop_max_ms": fields[53],
        "manipulability": fields[54],
        "gripper_present": fields[55],
        "gripper_activated": fields[56],
        "gripper_moving": fields[57],
        "gripper_object": fields[58],
        "gripper_position": fields[59],
        "gripper_target": fields[60],
        "gripper_current_A": fields[61],
        "fault": fault,
    }


def encode_event(name: str) -> bytes:
    return struct.pack("<4B", MAGIC, VERSION, TYPE_EVENT, EVENTS[name])


def encode_target(pose) -> bytes:
    return struct.pack("<4B7d", MAGIC, VERSION, TYPE_SET_TARGET, 0, *pose)


def encode_joint_velocities(qdot) -> bytes:
    return struct.pack("<3B6d", MAGIC, VERSION, TYPE_SET_JOINT_VELOCITIES, *qdot)


def encode_joint_positions(q) -> bytes:
    """Streamed absolute joint targets for the joint_position PD mode."""
    return struct.pack("<3B6d", MAGIC, VERSION, TYPE_SET_JOINT_POSITIONS, *q)


def encode_twist(twist) -> bytes:
    return struct.pack("<3B6d", MAGIC, VERSION, TYPE_SET_TWIST, *twist)


def encode_mode(mode: int) -> bytes:
    return struct.pack("<4B", MAGIC, VERSION, TYPE_SET_MODE, mode)


def encode_move_to_safe() -> bytes:
    """Payload-less; the safe pose is configured controller-side."""
    return struct.pack("<3B", MAGIC, VERSION, TYPE_MOVE_TO_SAFE)


def encode_zero_motor(motor: int) -> bytes:
    """Re-zero one joint's position datum (motor 1-6, standstill only)."""
    return struct.pack("<4B", MAGIC, VERSION, TYPE_ZERO_MOTOR, motor)


def encode_zero_all() -> bytes:
    """Zero motors 1-6 sequentially (server-side loop of ZERO_MOTOR)."""
    return struct.pack("<3B", MAGIC, VERSION, TYPE_ZERO_ALL)


def encode_gripper_activate() -> bytes:
    """Sequenced gripper activation (bring-up normally does this itself)."""
    return struct.pack("<3B", MAGIC, VERSION, TYPE_GRIPPER_ACTIVATE)


def encode_gripper_command(position: float, speed: float, force: float) -> bytes:
    """Move the gripper; all values normalized 0..1 (position 1 = closed)."""
    return struct.pack("<3B3f", MAGIC, VERSION, TYPE_GRIPPER_COMMAND,
                       position, speed, force)


def encode_gripper_stop() -> bytes:
    """Halt the gripper motion in place (never opens)."""
    return struct.pack("<3B", MAGIC, VERSION, TYPE_GRIPPER_STOP)


def encode_joint_velocities_ramped(qdot, accel) -> bytes:
    """Ramped jv: 6 targets (rad/s) + 6 accelerations (rad/s^2, > 0).

    The controller saturates accelerations at 0.5 rad/s^2 per joint.
    """
    return struct.pack("<3B12d", MAGIC, VERSION,
                       TYPE_SET_JOINT_VELOCITIES_RAMPED, *qdot, *accel)


def encode_ee_twist_ramped(twist, lin_accel: float, ang_accel: float) -> bytes:
    """Ramped tool twist: 6 targets + linear/angular accelerations (> 0).

    The controller saturates them at 1 m/s^2 linear, 0.5 rad/s^2 angular.
    """
    return struct.pack("<3B8d", MAGIC, VERSION, TYPE_SET_EE_TWIST_RAMPED,
                       *twist, lin_accel, ang_accel)


def encode_get_model() -> bytes:
    """Ask for the served kinematic model (answered with a MODEL frame)."""
    return struct.pack("<3B", MAGIC, VERSION, TYPE_GET_MODEL)


def parse_model(data: bytes):
    """MODEL frame -> dict(base_link, tip_link, urdf_xml bytes), or None."""
    if len(data) < 5:
        return None
    magic, version, ftype = struct.unpack_from("<3B", data)
    if magic != MAGIC or version != VERSION or ftype != TYPE_MODEL:
        return None
    pos = 3
    base_len = struct.unpack_from("<H", data, pos)[0]; pos += 2
    base_link = data[pos:pos + base_len].decode(); pos += base_len
    tip_len = struct.unpack_from("<H", data, pos)[0]; pos += 2
    tip_link = data[pos:pos + tip_len].decode(); pos += tip_len
    urdf_len = struct.unpack_from("<I", data, pos)[0]; pos += 4
    urdf = data[pos:pos + urdf_len]
    if len(urdf) != urdf_len:
        return None
    return {"base_link": base_link, "tip_link": tip_link, "urdf_xml": urdf}


def encode_get_target_frame() -> bytes:
    """Ask for the current and eligible tips (answered with TARGET_FRAME)."""
    return struct.pack("<3B", MAGIC, VERSION, TYPE_GET_TARGET_FRAME)


def parse_target_frame(data: bytes):
    """TARGET_FRAME frame -> dict(current, eligible list), or None."""
    if len(data) < 6:
        return None
    magic, version, ftype = struct.unpack_from("<3B", data)
    if magic != MAGIC or version != VERSION or ftype != TYPE_TARGET_FRAME:
        return None
    try:
        pos = 3
        n = struct.unpack_from("<H", data, pos)[0]; pos += 2
        current = data[pos:pos + n].decode(); pos += n
        count = data[pos]; pos += 1
        eligible = []
        for _ in range(count):
            n = struct.unpack_from("<H", data, pos)[0]; pos += 2
            eligible.append(data[pos:pos + n].decode()); pos += n
    except (struct.error, IndexError, UnicodeDecodeError):
        return None
    if pos != len(data):
        return None
    return {"current": current, "eligible": eligible}


def encode_set_target_frame(name: str) -> bytes:
    """Switch the controlled tip to a target_frame* link (answered with REPLY)."""
    raw = name.encode()
    return struct.pack("<3BH", MAGIC, VERSION, TYPE_SET_TARGET_FRAME, len(raw)) + raw


def encode_get_asset(name: str) -> bytes:
    """Ask for one URDF-referenced file (answered with an ASSET frame)."""
    raw = name.encode()
    return struct.pack("<3BH", MAGIC, VERSION, TYPE_GET_ASSET, len(raw)) + raw


def parse_asset(data: bytes):
    """ASSET frame -> (name, file bytes), or None (REPLY means rejected)."""
    if len(data) < 5:
        return None
    magic, version, ftype = struct.unpack_from("<3B", data)
    if magic != MAGIC or version != VERSION or ftype != TYPE_ASSET:
        return None
    pos = 3
    name_len = struct.unpack_from("<H", data, pos)[0]; pos += 2
    name = data[pos:pos + name_len].decode(); pos += name_len
    data_len = struct.unpack_from("<I", data, pos)[0]; pos += 4
    blob = data[pos:pos + data_len]
    if len(blob) != data_len:
        return None
    return name, blob


def encode_joint_target(q) -> bytes:
    """Planner: quintic move to an absolute joint target (rad, URDF order)."""
    return struct.pack("<3B6d", MAGIC, VERSION, TYPE_SET_JOINT_TARGET, *q)


def encode_pose_target(pose) -> bytes:
    """Planner: Cartesian target (execution pending singularity handling)."""
    return struct.pack("<3B7d", MAGIC, VERSION, TYPE_SET_POSE_TARGET, *pose)


def encode_joint_trajectory(waypoints, times=None, speed_scale: float = 1.0) -> bytes:
    """Planner: follow joint waypoints (rad, URDF order) continuously.

    Without ``times`` the planner fits a smooth path through the points and
    times it at its own limits; with ``times`` (seconds from the start, one
    per waypoint, strictly increasing) it follows that timing, stretched
    only if it would exceed the limits. ``speed_scale`` in (0, 1] slows
    either. The first waypoint must be (near) the current configuration.
    """
    rows = [tuple(float(v) for v in q) for q in waypoints]
    if not 1 <= len(rows) <= MAX_TRAJECTORY_WAYPOINTS:
        raise ValueError(f"1..{MAX_TRAJECTORY_WAYPOINTS} waypoints required")
    if any(len(q) != 6 for q in rows):
        raise ValueError("each waypoint needs 6 joint values")
    if times is not None and len(times) != len(rows):
        raise ValueError("one time per waypoint required")
    out = [struct.pack("<3BHBd", MAGIC, VERSION, TYPE_SET_JOINT_TRAJECTORY,
                       len(rows), 1 if times is not None else 0, float(speed_scale))]
    for i, q in enumerate(rows):
        out.append(struct.pack("<7d", float(times[i]) if times is not None else 0.0, *q))
    return b"".join(out)


def encode_cancel_target() -> bytes:
    """Planner: drop the active goal (arm holds via the deadman)."""
    return struct.pack("<3B", MAGIC, VERSION, TYPE_CANCEL_TARGET)


def encode_get_planner_status() -> bytes:
    """Planner: ask for a PLANNER_STATUS frame."""
    return struct.pack("<3B", MAGIC, VERSION, TYPE_GET_PLANNER_STATUS)


PLANNER_STATUS_FIXED = struct.Struct("<3B B 18d 5d I 3B 4d I H")


def parse_planner_status(data: bytes):
    """PLANNER_STATUS frame -> dict, or None (REPLY means no planner)."""
    if len(data) < PLANNER_STATUS_FIXED.size:
        return None
    fields = PLANNER_STATUS_FIXED.unpack_from(data)
    if fields[0] != MAGIC or fields[1] != VERSION or fields[2] != TYPE_PLANNER_STATUS:
        return None
    pair_len = fields[-1]
    pos = PLANNER_STATUS_FIXED.size
    pair = data[pos:pos + pair_len].decode("utf-8", "replace")
    pos += pair_len
    fault_len = struct.unpack_from("<H", data, pos)[0]
    pos += 2
    fault = data[pos:pos + fault_len].decode("utf-8", "replace")
    doubles = fields[4:22]
    return {
        "planner_state": fields[3],
        "target": doubles[0:6],
        "q_ref": doubles[6:12],
        "qd_ref": doubles[12:18],
        "time_remaining": fields[22],
        "max_tracking_error": fields[23],
        "loop_last_ms": fields[24],
        "loop_max_ms": fields[25],
        "loop_mean_ms": fields[26],
        "loop_overruns": fields[27],
        "collision_enabled": fields[28],
        "collision_approaching": fields[29],
        "collision_action": fields[30],
        "collision_min_distance": fields[31],
        "collision_sweep_last_ms": fields[32],
        "collision_sweep_max_ms": fields[33],
        "collision_sweep_mean_ms": fields[34],
        "collision_overruns": fields[35],
        "collision_pair": pair,
        "fault": fault,
    }


def parse_reply(data: bytes):
    if len(data) < 7:
        return None
    magic, version, ftype, status, state, msg_len = struct.unpack_from("<5BH", data)
    if magic != MAGIC or version != VERSION or ftype != TYPE_REPLY:
        return None
    message = data[7:7 + msg_len].decode("utf-8", "replace")
    return {"status": status, "controller_state": state, "message": message}


def connect_req(endpoint: str):
    """REQ socket configured with the timeouts every tool uses."""
    req = zmq.Context.instance().socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 3000)
    req.setsockopt(zmq.SNDTIMEO, 1000)
    req.setsockopt(zmq.LINGER, 0)
    req.connect(endpoint)
    return req


class Shell:
    """Latest-only telemetry subscriber plus a command request socket."""

    def __init__(self, pub_endpoint: str, cmd_endpoint: str):
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)  # latest-only
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.connect(pub_endpoint)
        self.req = connect_req(cmd_endpoint)

    def telemetry(self, timeout_s: float = 1.0):
        if self.sub.poll(int(timeout_s * 1000)) == 0:
            return None
        return parse_telemetry(self.sub.recv())

    def request(self, frame: bytes):
        self.req.send(frame)
        try:
            return parse_reply(self.req.recv())
        except zmq.error.Again:
            return None

    def send_event(self, name: str):
        reply = self.request(encode_event(name))
        if reply is None:
            print(f"event {name}: NO REPLY (server down?)")
            return False
        print(f"event {name}: {REPLY_STATUS.get(reply['status'])} "
              f"state={CONTROLLER_STATES.get(reply['controller_state'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_target(self, pose):
        reply = self.request(encode_target(pose))
        if reply is None:
            print("target: NO REPLY")
            return False
        print(f"target: {REPLY_STATUS.get(reply['status'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_mode(self, mode_name: str):
        code = MODE_CODES.get(mode_name)
        if code is None:
            print(f"unknown mode: {mode_name} (pose|jv|world_twist|ee_twist)")
            return False
        reply = self.request(encode_mode(code))
        if reply is None:
            print("mode: NO REPLY")
            return False
        print(f"mode: {REPLY_STATUS.get(reply['status'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_joint_target(self, q):
        """Planner quintic move; UNSUPPORTED against a bare controller."""
        reply = self.request(encode_joint_target(q))
        if reply is None:
            print("moveto: NO REPLY")
            return False
        print(f"moveto: {REPLY_STATUS.get(reply['status'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_joint_trajectory(self, waypoints, times=None, speed_scale: float = 1.0,
                              quiet: bool = False):
        """Planner waypoint follower; the REPLY dict, or None (no reply)."""
        reply = self.request(encode_joint_trajectory(waypoints, times, speed_scale))
        if not quiet:
            if reply is None:
                print("trajectory: NO REPLY")
            else:
                print(f"trajectory: {REPLY_STATUS.get(reply['status'])}"
                      + (f" ({reply['message']})" if reply["message"] else ""))
        return reply

    def send_cancel_target(self):
        reply = self.request(encode_cancel_target())
        if reply is None:
            print("cancel: NO REPLY")
            return False
        print(f"cancel: {REPLY_STATUS.get(reply['status'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def planner_status(self):
        """PLANNER_STATUS dict, or None (bare controller / no reply)."""
        self.req.send(encode_get_planner_status())
        try:
            data = self.req.recv()
        except zmq.error.Again:
            print("pstatus: NO REPLY")
            return None
        status = parse_planner_status(data)
        if status is not None:
            return status
        reply = parse_reply(data)
        if reply is not None:
            print(f"pstatus: {REPLY_STATUS.get(reply['status'])}"
                  + (f" ({reply['message']})" if reply["message"] else ""))
        return None

    def get_target_frame(self):
        """dict(current, eligible), or None (old server / no reply)."""
        self.req.send(encode_get_target_frame())
        try:
            data = self.req.recv()
        except zmq.error.Again:
            print("tip: NO REPLY")
            return None
        tips = parse_target_frame(data)
        if tips is not None:
            return tips
        reply = parse_reply(data)
        if reply is not None:
            print(f"tip: {REPLY_STATUS.get(reply['status'])}"
                  + (f" ({reply['message']})" if reply["message"] else ""))
        return None

    def set_target_frame(self, name: str):
        """Switch the controlled tip; the arm must be stationary."""
        reply = self.request(encode_set_target_frame(name))
        if reply is None:
            print(f"tip {name}: NO REPLY")
            return False
        print(f"tip {name}: {REPLY_STATUS.get(reply['status'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_move_to_safe(self):
        """Sequence run/jv-switch/P-converge to the controller's safe pose."""
        reply = self.request(encode_move_to_safe())
        if reply is None:
            print("move_to_safe: NO REPLY")
            return False
        print(f"move_to_safe: {REPLY_STATUS.get(reply['status'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_zero_motor(self, motor: int):
        """Zero one joint's datum; completion is deferred (poll q/fault)."""
        reply = self.request(encode_zero_motor(motor))
        if reply is None:
            print(f"zero {motor}: NO REPLY")
            return False
        print(f"zero {motor}: {REPLY_STATUS.get(reply['status'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_zero_all_motors(self):
        """Zero motors 1-6 sequentially; completion is deferred (~6 s max)."""
        reply = self.request(encode_zero_all())
        if reply is None:
            print("zero all: NO REPLY")
            return False
        print(f"zero all: {REPLY_STATUS.get(reply['status'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_gripper_activate(self):
        """Sequenced gripper activation; poll gripper_activated in telemetry."""
        reply = self.request(encode_gripper_activate())
        if reply is None:
            print("gripper activate: NO REPLY")
            return False
        print(f"gripper activate: {REPLY_STATUS.get(reply['status'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_gripper_command(self, position: float, speed: float = 1.0,
                             force: float = 0.25, quiet: bool = False):
        """Move the gripper (0..1 values); completion via telemetry."""
        reply = self.request(encode_gripper_command(position, speed, force))
        if reply is None:
            print("gripper: NO REPLY")
            return False
        if not quiet or reply["status"] != 0:
            print(f"gripper: {REPLY_STATUS.get(reply['status'])}"
                  + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_gripper_stop(self):
        """Halt the gripper motion in place."""
        reply = self.request(encode_gripper_stop())
        if reply is None:
            print("gripper stop: NO REPLY")
            return False
        print(f"gripper stop: {REPLY_STATUS.get(reply['status'])}"
              + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_jv(self, qdot, quiet: bool = False):
        reply = self.request(encode_joint_velocities(qdot))
        if reply is None:
            print("jv: NO REPLY")
            return False
        if not quiet or reply["status"] != 0:
            print(f"jv: {REPLY_STATUS.get(reply['status'])}"
                  + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_jp(self, q, quiet: bool = False):
        """Absolute joint targets; joint_position (PD) mode only. Streamed:
        the 200 ms deadman halts the arm when targets stop arriving."""
        reply = self.request(encode_joint_positions(q))
        if reply is None:
            print("jp: NO REPLY")
            return False
        if not quiet or reply["status"] != 0:
            print(f"jp: {REPLY_STATUS.get(reply['status'])}"
                  + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_twist(self, twist, quiet: bool = False):
        reply = self.request(encode_twist(twist))
        if reply is None:
            print("twist: NO REPLY")
            return False
        if not quiet or reply["status"] != 0:
            print(f"twist: {REPLY_STATUS.get(reply['status'])}"
                  + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_jvr(self, qdot, accel, quiet: bool = False):
        """Ramped jv input; accel is one value per joint (rad/s^2)."""
        reply = self.request(encode_joint_velocities_ramped(qdot, accel))
        if reply is None:
            print("jvr: NO REPLY")
            return False
        if not quiet or reply["status"] != 0:
            print(f"jvr: {REPLY_STATUS.get(reply['status'])}"
                  + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def send_eevr(self, twist, lin_accel: float, ang_accel: float,
                  quiet: bool = False):
        """Ramped tool-twist input with linear/angular accelerations."""
        reply = self.request(encode_ee_twist_ramped(twist, lin_accel, ang_accel))
        if reply is None:
            print("eevr: NO REPLY")
            return False
        if not quiet or reply["status"] != 0:
            print(f"eevr: {REPLY_STATUS.get(reply['status'])}"
                  + (f" ({reply['message']})" if reply["message"] else ""))
        return reply["status"] == 0

    def stream(self, kind: str, values, seconds: float, rate_hz: float = 50.0,
               accel=None):
        """Refresh a streamed-mode input for a duration (deadman is 200 ms).

        kind is one of jv, jp, twist, jvr, eevr; ramped kinds require accel
        (a 6-tuple for jvr, a (lin, ang) pair for eevr).
        """
        if kind == "jv":
            def send(v, quiet):
                return self.send_jv(v, quiet=quiet)
        elif kind == "jp":
            def send(v, quiet):
                return self.send_jp(v, quiet=quiet)
        elif kind == "twist":
            def send(v, quiet):
                return self.send_twist(v, quiet=quiet)
        elif kind == "jvr":
            def send(v, quiet):
                return self.send_jvr(v, accel, quiet=quiet)
        else:
            def send(v, quiet):
                return self.send_eevr(v, accel[0], accel[1], quiet=quiet)
        deadline = time.monotonic() + seconds
        ok = True
        first = True
        while time.monotonic() < deadline and ok:
            ok = send(values, quiet=not first)
            first = False
            time.sleep(1.0 / rate_hz)
        return ok

    def wait_state(self, state_name: str, timeout_s: float,
                   require_motion: bool = False) -> bool:
        want = {v: k for k, v in CONTROLLER_STATES.items()}[state_name]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            t = self.telemetry(0.5)
            if t and t["controller_state"] == want and \
                    (not require_motion or t["motion_active"]):
                return True
        return False

    def wait_session(self, session_name: str, timeout_s: float) -> bool:
        """Wait for the serial session state (e.g. Held = sequence complete)."""
        want = {v: k for k, v in SESSION_STATES.items()}[session_name]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            t = self.telemetry(0.5)
            if t and t["session_state"] == want:
                return True
        return False

    def wait_mode(self, mode_name: str, timeout_s: float) -> bool:
        """Wait until the control mode is active and no transition settles."""
        want = MODE_CODES.get(mode_name)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            t = self.telemetry(0.5)
            if t and t["control_mode"] == want and not t["settling"]:
                return True
        return False

    def status_line(self, t) -> str:
        if t is None:
            return "no telemetry (server down?)"
        q = " ".join(f"{v:+.3f}" for v in t["q"])
        settling = " settling" if t["settling"] else ""
        age = (f" input_age={t['input_age_ms']:.0f}ms"
               if t["input_age_ms"] >= 0 else "")
        gripper = ""
        if t["gripper_present"]:
            flags = "act" if t["gripper_activated"] else "inact"
            if t["gripper_moving"]:
                flags += ",moving"
            if t["gripper_object"]:
                flags += ",object"
            gripper = f" grip={t['gripper_position']:.2f}({flags})"
        manip = (f" manip={t['manipulability']:.4f}"
                 if t.get("manipulability", -1.0) >= 0 else "")
        return (f"seq={t['sequence']} "
                f"ctrl={CONTROLLER_STATES.get(t['controller_state'])} "
                f"mode={CONTROL_MODES.get(t['control_mode'])}{settling} "
                f"session={SESSION_STATES.get(t['session_state'])} "
                f"fw={FIRMWARE_STATES.get(t['firmware_state'])}/{t['firmware_reason']:#04x} "
                f"active={t['motion_active']}{age}{manip} q=[{q}]{gripper}"
                + (f" fault=\"{t['fault']}\"" if t["fault"] else ""))

    def measure_rate(self, seconds: float) -> float:
        # CONFLATE keeps only the newest frame; count via sequence numbers.
        first = last = None
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            t = self.telemetry(0.5)
            if t is None:
                continue
            if first is None:
                first = (time.monotonic(), t["sequence"])
            last = (time.monotonic(), t["sequence"])
        if first is None or last is None or last[1] == first[1]:
            return 0.0
        return (last[1] - first[1]) / (last[0] - first[0])
