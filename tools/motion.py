#!/usr/bin/env python3
"""Closed-loop motion helpers built on the robot_interface Shell.

State sequencing (ensure_running_mode), a per-joint PID mover in
joint_velocity mode (move_to_joints), and a Cartesian pose servo in
world_twist mode (servo_to_pose); the mover scripts in this directory are
thin CLIs over these. Both loops stream at LOOP_RATE_HZ, well inside
the controller's 200 ms input deadman, and always end by commanding
zeros. The controller additionally clamps joint velocities to 2.5 rad/s
each, so the effective ceiling is the lower of its clamp and the ones
here.
"""

import math
import time

MAX_JOINT_VEL = 10.0     # rad/s, clamp on move_to_joints output
MAX_LIN_VEL = 1.0        # m/s, clamp on the servo_to_pose linear twist
MAX_ANG_VEL = 0.5        # rad/s, clamp on the servo_to_pose angular twist
LOOP_RATE_HZ = 50.0      # must stay well inside the controller's 200 ms deadman
SETTLED_CYCLES = 25      # consecutive in-tolerance cycles required to finish
STALL_TELEMETRY_S = 0.5  # abort if telemetry stops for this long


def add_joint_pid_arguments(parser):
    """Gain and termination flags shared by the joint-space movers."""
    parser.add_argument("--kp", type=float, default=7.0)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.005)
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="per-joint position tolerance, rad (default 0.005)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="abort after this many seconds (default 60)")


def add_pose_servo_arguments(parser):
    """Gain and termination flags shared by the twist-servo movers."""
    parser.add_argument("--kp-lin", type=float, default=1.5)
    parser.add_argument("--kp-ang", type=float, default=1.0)
    parser.add_argument("--tol-lin", type=float, default=0.002,
                        help="position tolerance, m (default 0.002)")
    parser.add_argument("--tol-ang", type=float, default=0.005,
                        help="orientation tolerance, rad (default 0.005)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="abort after this many seconds (default 60)")


class JointPid:
    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.integral = 0.0
        self.last_error = None

    def step(self, error: float) -> float:
        self.integral += error * self.dt
        # Anti-windup: keep the integral term alone within the velocity clamp.
        if self.ki > 0:
            limit = MAX_JOINT_VEL / self.ki
            self.integral = max(-limit, min(limit, self.integral))
        derivative = 0.0
        if self.last_error is not None:
            derivative = (error - self.last_error) / self.dt
        self.last_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-MAX_JOINT_VEL, min(MAX_JOINT_VEL, out))


def quat_conj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotation_error(q_current, q_target):
    """Axis-angle rotation (rad, base frame) taking current to target."""
    ex, ey, ez, ew = quat_mul(q_target, quat_conj(q_current))
    if ew < 0.0:  # shorter arc
        ex, ey, ez, ew = -ex, -ey, -ez, -ew
    norm = math.sqrt(ex * ex + ey * ey + ez * ez)
    if norm < 1e-9:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(norm, ew)
    return (ex / norm * angle, ey / norm * angle, ez / norm * angle)


def clamp_norm(vec, limit):
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= limit or norm < 1e-12:
        return vec
    scale = limit / norm
    return tuple(v * scale for v in vec)


def rpy_to_quat(roll, pitch, yaw):
    """ZYX roll/pitch/yaw (rad) -> (qx, qy, qz, qw); the controller's and
    the jogger's convention (tcp_rpy_deg, move_pose rpy_deg)."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def pose_servo_step(tcp, target_p, target_q, kp_lin, kp_ang, v_max, w_max):
    """One step of the base-frame pose P-servo: (twist, pos_err, rot_err).

    ``tcp`` is the telemetry tcp_pose (x y z qx qy qz qw); the twist is
    the SET_TWIST payload for world_twist mode, linear and angular parts
    clamped to ``v_max`` (m/s) / ``w_max`` (rad/s); the errors are the
    remaining distance (m) and rotation (rad). servo_to_pose() below is
    the blocking loop over this step; teleop_quest.py runs it one tick at
    a time inside its own control loop (random start poses).
    """
    e_p = tuple(ref - cur for ref, cur in zip(target_p, tcp[0:3]))
    e_r = rotation_error(tuple(tcp[3:7]), target_q)
    lin = clamp_norm(tuple(kp_lin * e for e in e_p), v_max)
    ang = clamp_norm(tuple(kp_ang * e for e in e_r), w_max)
    pos_err = math.sqrt(sum(v * v for v in e_p))
    rot_err = math.sqrt(sum(v * v for v in e_r))
    return tuple(lin) + tuple(ang), pos_err, rot_err


# ------------------------------------------------------------ start points
#
# Randomized episode start poses for data collection (teleop_quest.py
# --start-points): a JSON file with a few base-frame TCP poses, x y z
# roll pitch yaw each, sampled per episode either anywhere in the range
# they span ("box": every coordinate uniform between the min and max of
# the points, angles included) or as one of the points ("pick"). Keys
# starting with "_" are comments.
#
#   {"units": "mm", "mode": "box",
#    "points": [[450, -150, 300, 180, 0, 90],
#               [550,  150, 300, 180, 0, 90],
#               [500,    0, 400, 170, 10, 80]],
#    "speed": 0.25, "ang_speed": 0.5, "tol_lin": 0.005, "tol_ang": 0.02,
#    "timeout_s": 30}
#
# points   x y z roll pitch yaw: position in `units`, angles in degrees
#          (ZYX roll/pitch/yaw, the jogger's Orientation readout). Angle
#          ranges are taken the short way round, so 170 .. -170 spans 20
#          degrees through 180, not 340. Points with only x y z are
#          accepted too: the orientation is then held (rpy_deg if given,
#          else the one the arm has when the move starts).
# units    "m" (default) or "mm" for the positions (the jogger shows mm).
# via_safe false (default): servo straight from wherever the arm is to
#          the sampled pose; true: MOVE_TO_SAFE first, so every approach
#          starts from the same joint configuration.
# speed / ang_speed   clamps on the servo twist (m/s, rad/s).
# tol_lin / tol_ang   arrival tolerances (m, rad); timeout_s per move.

START_POINTS_DEFAULTS = {
    "mode": "box", "via_safe": False, "speed": 0.25, "ang_speed": 0.5,
    "kp_lin": 1.5, "kp_ang": 1.0, "tol_lin": 0.005, "tol_ang": 0.02,
    "timeout_s": 30.0,
}


def load_start_points(path) -> dict:
    """Parse and validate a start-points file; SystemExit on a bad one.

    Returns START_POINTS_DEFAULTS overlaid with the file, points in
    meters / radians as tuples of 3 or 6 floats (all the same length),
    ``rpy`` (rad) or None, and ``path``.
    """
    import json
    from pathlib import Path

    path = Path(path).expanduser()
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"start points {path}: {exc}") from None
    if not isinstance(doc, dict):
        raise SystemExit(f"start points {path}: expected a JSON object")
    cfg = dict(START_POINTS_DEFAULTS)
    cfg.update({k: v for k, v in doc.items() if not str(k).startswith("_")})
    cfg["path"] = str(path)

    units = str(cfg.pop("units", "m")).lower()
    if units not in ("m", "mm"):
        raise SystemExit(f"start points {path}: units must be m or mm, got {units!r}")
    scale = 0.001 if units == "mm" else 1.0
    if cfg["mode"] not in ("box", "pick"):
        raise SystemExit(f"start points {path}: mode must be box or pick, got {cfg['mode']!r}")
    raw = cfg.get("points")
    if not isinstance(raw, list) or len(raw) < 1:
        raise SystemExit(f"start points {path}: 'points' must be a non-empty list")
    points = []
    for i, p in enumerate(raw):
        if not isinstance(p, list) or len(p) not in (3, 6):
            raise SystemExit(f"start points {path}: point {i} must have 3 (x y z) or "
                             f"6 (x y z roll pitch yaw) numbers, got {p!r}")
        try:
            vals = [float(v) for v in p]
        except (TypeError, ValueError):
            raise SystemExit(f"start points {path}: point {i} is not numeric: {p!r}") from None
        if not all(math.isfinite(v) for v in vals):
            raise SystemExit(f"start points {path}: point {i} is not finite: {p!r}")
        pos = [v * scale for v in vals[:3]]
        ang = [math.radians(v) for v in vals[3:]]
        points.append(tuple(pos + ang))
    if len({len(p) for p in points}) != 1:
        raise SystemExit(f"start points {path}: mix of 3- and 6-number points")
    cfg["points"] = points
    rpy = cfg.pop("rpy_deg", None)
    if rpy is not None:
        if len(points[0]) == 6:
            raise SystemExit(f"start points {path}: rpy_deg is for 3-number points; "
                             "6-number points carry their own orientation")
        try:
            rpy = tuple(math.radians(float(v)) for v in rpy)
        except (TypeError, ValueError):
            raise SystemExit(f"start points {path}: rpy_deg must be 3 numbers") from None
        if len(rpy) != 3:
            raise SystemExit(f"start points {path}: rpy_deg must be 3 numbers")
    cfg["rpy"] = rpy
    for key in ("speed", "ang_speed", "kp_lin", "kp_ang", "tol_lin", "tol_ang", "timeout_s"):
        try:
            cfg[key] = float(cfg[key])
        except (TypeError, ValueError):
            raise SystemExit(f"start points {path}: {key} must be a number") from None
        if not cfg[key] > 0.0:
            raise SystemExit(f"start points {path}: {key} must be positive")
    cfg["speed"] = min(cfg["speed"], MAX_LIN_VEL)
    cfg["ang_speed"] = min(cfg["ang_speed"], MAX_ANG_VEL)
    cfg["via_safe"] = bool(cfg["via_safe"])
    return cfg


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def sample_start_pose(cfg: dict, rng=None):
    """One start pose from a load_start_points() config: (position m,
    rpy rad or None). "box" draws each coordinate uniformly between the
    min and max over the points -- the three angles too, measured the
    short way round from the first point so a range through +-180 deg
    stays narrow; "pick" returns one of the points."""
    import random

    rng = rng or random
    points = cfg["points"]
    if cfg["mode"] == "pick":
        chosen = tuple(rng.choice(points))
    else:
        chosen = []
        for i in range(len(points[0])):
            vals = [p[i] for p in points]
            if i >= 3:  # angles: unwrap onto the first point's branch
                vals = [points[0][i] + _wrap_angle(v - points[0][i]) for v in vals]
            v = rng.uniform(min(vals), max(vals))
            chosen.append(_wrap_angle(v) if i >= 3 else v)
        chosen = tuple(chosen)
    rpy = tuple(chosen[3:6]) if len(chosen) == 6 else cfg.get("rpy")
    return tuple(chosen[:3]), rpy


def describe_start_pose(position, rpy) -> str:
    text = "[%.3f %.3f %.3f] m" % tuple(position)
    if rpy is not None:
        text += " rpy [%.0f %.0f %.0f] deg" % tuple(math.degrees(a) for a in rpy)
    return text


def wait_held_or_sim(shell, timeout_s: float) -> bool:
    """Ready when the controller is power_on and, on hardware, the serial
    session reached Held (the simulator has no session and reports 0)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        t = shell.telemetry(0.5)
        if t and t["controller_state"] == 1 and t["session_state"] in (0, 3):
            return True
    return False


def ensure_running_mode(shell, mode_name: str = "jv") -> bool:
    """Drive the controller to running + motion active in the given mode."""
    t = shell.telemetry()
    if t is None:
        print("no telemetry; is the controller serving?")
        return False

    if t["controller_state"] == 3:
        print("controller in error; recovering")
        if not shell.send_event("clear") or not wait_held_or_sim(shell, 45.0):
            return False

    t = shell.telemetry()
    if t["controller_state"] == 0:
        if not shell.send_event("power_on"):
            return False
        if not wait_held_or_sim(shell, 45.0):
            print("power on did not reach HOLD")
            return False

    # Select the mode; while running this is a guarded switch (settle to
    # standstill, clear inputs, flip), otherwise a preselect.
    if not shell.send_mode(mode_name):
        return False

    t = shell.telemetry()
    if t["controller_state"] != 2:
        # The facade sequencer may still be finishing POWER_ON (busy) for a
        # moment after the state flips; retry until RUN is accepted.
        for _ in range(20):
            if shell.send_event("run"):
                break
            time.sleep(0.25)
        else:
            print("run was never accepted")
            return False
        if not shell.wait_state("running", 15.0):
            print("run did not reach running")
            return False
    if not shell.wait_mode(mode_name, 10.0):
        print(f"mode did not settle to {mode_name}")
        return False

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        t = shell.telemetry()
        if t and t["motion_active"]:
            return True
    print("motion did not become active")
    return False


def move_to_joints(shell, target, kp: float, ki: float, kd: float,
                   tolerance: float, timeout: float) -> int:
    """PID-drive the joints to `target` (rad). Returns 0 on convergence.

    Preconditions: running with motion active in joint_velocity mode (see
    ensure_running_mode). Always ends by commanding zeros; Ctrl-C sends
    zeros and exits.
    """
    t = shell.telemetry()
    start_q = " ".join(f"{v:+.3f}" for v in t["q"])
    target_str = " ".join(f"{v:+.3f}" for v in target)
    print(f"start q = [{start_q}], target [{target_str}] rad "
          f"(kp={kp} ki={ki} kd={kd}, clamp {MAX_JOINT_VEL} rad/s)")

    dt = 1.0 / LOOP_RATE_HZ
    pids = [JointPid(kp, ki, kd, dt) for _ in range(6)]
    settled = 0
    last_telemetry = time.monotonic()
    deadline = time.monotonic() + timeout
    next_tick = time.monotonic()
    result = 1

    try:
        while time.monotonic() < deadline:
            next_tick += dt
            t = shell.telemetry(timeout_s=0.1)
            now = time.monotonic()
            if t is None:
                if now - last_telemetry > STALL_TELEMETRY_S:
                    print("telemetry stalled; aborting")
                    break
                continue
            last_telemetry = now

            if t["controller_state"] != 2 or not t["motion_active"] or t["fault"]:
                print(f"controller left motion state: {shell.status_line(t)}")
                break

            errors = [ref - q for ref, q in zip(target, t["q"])]
            command = tuple(pid.step(e) for pid, e in zip(pids, errors))

            max_err = max(abs(e) for e in errors)
            if max_err < tolerance:
                settled += 1
                if settled >= SETTLED_CYCLES:
                    shell.send_jv((0.0,) * 6, quiet=True)
                    final_q = " ".join(f"{v:+.4f}" for v in t["q"])
                    print(f"done: q = [{final_q}], max error "
                          f"{max_err * 1000:.2f} mrad")
                    result = 0
                    break
            else:
                settled = 0

            if not shell.send_jv(command, quiet=True):
                print("velocity command rejected; aborting")
                break

            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
        else:
            print(f"timeout after {timeout:.0f} s; sending zeros")
    except KeyboardInterrupt:
        print("\ninterrupted; sending zeros")
    finally:
        try:
            shell.send_jv((0.0,) * 6, quiet=True)
        except Exception:
            pass

    return result


def servo_to_pose(shell, target_p, target_q, kp_lin, kp_ang,
                  tol_lin, tol_ang, timeout):
    """Stream base-frame twist commands until the TCP reaches the pose.

    Returns 0 on convergence, 1 otherwise. Preconditions: running with
    motion active in world_twist mode (see ensure_running_mode). Always
    ends by commanding a zero twist; Ctrl-C sends zeros and exits.
    """
    dt = 1.0 / LOOP_RATE_HZ
    settled = 0
    last_telemetry = time.monotonic()
    deadline = time.monotonic() + timeout
    next_tick = time.monotonic()
    result = 1

    try:
        while time.monotonic() < deadline:
            next_tick += dt
            t = shell.telemetry(timeout_s=0.1)
            now = time.monotonic()
            if t is None:
                if now - last_telemetry > STALL_TELEMETRY_S:
                    print("telemetry stalled; aborting")
                    break
                continue
            last_telemetry = now

            if t["controller_state"] != 2 or not t["motion_active"] or t["fault"]:
                print(f"controller left motion state: {shell.status_line(t)}")
                break

            tcp = t["tcp_pose"]
            twist, pos_err, rot_err = pose_servo_step(
                tcp, target_p, target_q, kp_lin, kp_ang, MAX_LIN_VEL, MAX_ANG_VEL)
            if pos_err < tol_lin and rot_err < tol_ang:
                settled += 1
                if settled >= SETTLED_CYCLES:
                    shell.send_twist((0.0,) * 6, quiet=True)
                    print(f"done: tcp = [{tcp[0]:+.4f} {tcp[1]:+.4f} "
                          f"{tcp[2]:+.4f}] m, position error "
                          f"{pos_err * 1000:.2f} mm, orientation error "
                          f"{rot_err * 1000:.2f} mrad")
                    result = 0
                    break
            else:
                settled = 0

            if not shell.send_twist(twist, quiet=True):
                print("twist command rejected; aborting")
                break

            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
        else:
            print(f"timeout after {timeout:.0f} s; sending zeros")
    except KeyboardInterrupt:
        print("\ninterrupted; sending zeros")
    finally:
        try:
            shell.send_twist((0.0,) * 6, quiet=True)
        except Exception:
            pass

    return result
