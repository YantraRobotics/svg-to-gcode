"""Build a clean MJCF for the OWL 6.9 arm from the vendor mesh geometry.

Why not convert the URDF directly? The vendor export keeps every STL in one
shared, fully-assembled CAD frame and then tries to cancel the link frames
with per-visual offsets. Those offsets are corrupt for the wrist chain
(joint anchors up to 1.75 m away from the physical motors) and MuJoCo's
URDF import additionally drops one fixed-link offset, so IK on the
converted model is meaningless.

The meshes themselves are trustworthy, and they cross-validate against the
URDF's own link lengths (elbow motor sits exactly 0.291 m above the
shoulder motor, matching the URDF's upper-arm offset). This script:

1. reads the vendor drop checked in at robots/ (URDF_hand_e_aruco.urdf +
   the STL meshes -- the same files the C++ controller loads),
2. measures each motor-housing STL to place the joint axes, after moving
   the meshes from the shoulder bracket on by the URDF's base-stack
   correction (base_stack_shift; the CAD seats the bracket 35 mm too high),
3. reads only the joint AXIS DIRECTIONS from the URDF for the arm
   (trusted: the vendor flipped these deliberately in the latest export),
4. places the Robotiq Hand-E and the "tcp"/"target" sites exactly where
   the URDF puts them: the hand_e_mount fixed joint plus the hand link's
   visual origins are composed by forward kinematics at the zero pose
   (no hand-crafted mount transform; an axis-aligned guess previously
   rolled the hand 90 degrees against the real robot),
5. self-validates: sweeping any joint must not displace its own motor
   housing (a wrong anchor shows up as the housing orbiting the axis).

Output (git-ignored, rebuilt on demand), consumed by
firmware_emulator.py --mujoco:

    tools/owl69_mjcf/owl69.xml
    tools/owl69_mjcf/assets/*.stl

Run from the tools uv environment:

    uv run python prepare_owl69_model.py

Or build from the model a running controller serves over ZMQ (GET_MODEL
plus GET_ASSET, see PROTOCOL.md), so the sim inherits exactly the URDF
and meshes the controller runs with:

    uv run python prepare_owl69_model.py --from-controller tcp://127.0.0.1:5556
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
# The same URDF the controller loads by default (Config::urdf_path in
# include/robot_controller/controller_runtime.hpp); it includes the Hand-E
# with the ArUco cube and D405, and the target_frame tip.
DEFAULT_URDF = HERE.parent / "robots" / "OWL_Release" / "URDF_hand_e_aruco.urdf"
BUNDLE_DIR = HERE / "owl69_mjcf"
MODEL_XML = BUNDLE_DIR / "owl69.xml"
#: Fetch mode mirrors the controller-served URDF and its mesh assets here.
#: The URDF sits one level down (served/robot/model.urdf) so its relative
#: "../meshes/..." references resolve inside served/.
SERVED_DIR = BUNDLE_DIR / "served"

MESH_SCALE = 0.001  # vendor STLs are in millimetres

#: Joint chain in order. Each joint's axis passes through the bounding-box
#: centre of its motor-housing mesh (AD_80110 / AD_50070 actuators); the
#: URDF supplies only the axis direction. Anchor components along the axis
#: are irrelevant, perpendicular ones are what the broken URDF got wrong.
CHAIN = (
    # (joint name, motor-housing mesh, meshes that move with the joint)
    (
        "joint1_base_yaw",
        "AD_80110_Base_v2_1.stl",
        ["AD_80110_Base_v2_1.stl", "ShoulderJoint_v1_1.stl"],
    ),
    (
        "joint2_shoulder",
        "AD_80110_J2_v2_1.stl",
        ["AD_80110_J2_v2_1.stl", "Link1_urdf_v1_1.stl", "AD_80110_J3_v1_1.stl"],
    ),
    (
        "joint3_elbow",
        "AD_80110_J3_v1_1.stl",
        ["Link2_urdf_v1_1.stl", "AD_50070_J4_v2_1.stl"],
    ),
    (
        "joint4_wrist_roll",
        "AD_50070_J4_v2_1.stl",
        ["Wrist_Link_urdf_v1_1.stl", "AD_50070_J4_v2_2.stl"],
    ),
    (
        "joint5_wrist_pitch",
        "AD_50070_J4_v2_2.stl",
        [
            "Wrist_Link_urdf_v1__1__1.stl",
            "AD_50070_J4_v2_3.stl",
            "5.17-681-ASM_5.17-681-U_WJ_P0_R0_v1_1__1__v1_1.stl",
        ],
    ),
    (
        "joint6_ee_rotate",
        "AD_50070_J4_v2_3.stl",
        ["6.8_URDF_end_effector_v1_1__1__v1_1.stl"],
    ),
)

BASE_MESH = "base_link.stl"

# Origin of the baked base_link.stl, expressed in the assembled CAD
# world the remaining arm STLs still live in. CAD_vertex - CAD_TO_BASE
# is that vertex in the URDF base_link frame.
CAD_TO_BASE = np.array([-0.116463665009, -0.013676971436, 0.111960067749])

# The vendor CAD seats the shoulder bracket on top of the base actuator's
# thick ring; on the robot its plate sits under that ring, and the URDF
# (rigid7) lowers the bracket and everything after it accordingly. The
# CAD-world meshes from the bracket on are moved by that same offset
# (base_stack_shift); the base actuator itself stays where the CAD has it.
BASE_ACTUATOR_MESH = "AD_80110_Base_v2_1.stl"
SHOULDER_LINK = "ShoulderJoint_v1_1"
SHOULDER_MESH = "ShoulderJoint_v1_1.stl"

# Pitch- and EE-rotate motor housings, plus the wrist adapter plate;
# vertices were baked into their URDF link frames (mounting / EE-face
# center).
J5_MESH = "AD_50070_J4_v2_2.stl"
J6_MESH = "AD_50070_J4_v2_3.stl"
ADAPTER_MESH = "5.17-681-ASM_5.17-681-U_WJ_P0_R0_v1_1__1__v1_1.stl"
LINK_FRAME_MESHES = (J5_MESH, J6_MESH, ADAPTER_MESH)

#: The URDF links that own the Hand-E geometry and the controller's tip
#: frame. Their zero-pose world transforms (composed by forward
#: kinematics over the URDF's fixed origins) place the hand body, its
#: mesh visuals, and the tcp site; nothing about the tool is guessed.
HAND_LINK = "hand_e_base"
TARGET_LINK = "target_frame"  # controller tip_link in controller_runtime.hpp

HAND_BODY = HAND_LINK      # rigid (jointless) child of the EE body
TARGET_SITE = "target"     # the hand link frame itself (see build_spec)

ARM_JOINTS = tuple(joint for joint, _, _ in CHAIN)
EE_BODY = "end_effector_link"  # body carrying joint6 + the gripper
TCP_SITE = "tcp"

SILVER = (0.72, 0.74, 0.77, 1.0)

#: Joint range (rad). The vendor URDF declares every joint "continuous"
#: with no limit tags; +/-175 deg keeps IK in one winding and is what the
#: real controller enforces in software.
JOINT_RANGE = 3.05

#: Drive parameters per joint, single source of truth for the model's
#: position actuators, the emulator's setpoint slewing, and the teleop
#: host's clamps. The URDF carries no limits, so velocity/effort are
#: conservative estimates from the actuator classes (AD_80110 on the
#: proximal three joints, AD_50070 on the wrist): tighten against the
#: vendor datasheet before driving real hardware.
#:   (max velocity rad/s, peak torque Nm, servo kp, servo kd, damping)
#: kd is tuned near critical damping so a slewed setpoint never drives a
#: joint past its velocity limit (pinned by tests/test_owl_emulator.py).
JOINT_DRIVES: dict[str, tuple[float, float, float, float, float]] = {
    "joint1_base_yaw": (2.0, 110.0, 600.0, 60.0, 3.0),
    "joint2_shoulder": (2.0, 110.0, 600.0, 60.0, 3.0),
    "joint3_elbow": (2.0, 110.0, 600.0, 60.0, 3.0),
    "joint4_wrist_roll": (3.5, 50.0, 150.0, 15.0, 1.0),
    "joint5_wrist_pitch": (3.5, 50.0, 150.0, 15.0, 1.0),
    "joint6_ee_rotate": (3.5, 50.0, 150.0, 15.0, 1.0),
}

VEL_LIMIT = np.array([JOINT_DRIVES[j][0] for j in ARM_JOINTS])


def vendor_urdf() -> Path:
    """The vendor URDF checked in under robots/OWL_Release/."""
    if DEFAULT_URDF.is_file():
        return DEFAULT_URDF
    raise FileNotFoundError(
        f"{DEFAULT_URDF} is missing; restore robots/OWL_Release/ and "
        "robots/meshes/OWL_Release/ (the vendor URDF + meshes) or pass --urdf"
    )


def _rpy_matrix(rpy) -> np.ndarray:
    """URDF rpy (extrinsic x-y-z) to a rotation matrix: Rz @ Ry @ Rx."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _origin(element) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of an optional URDF <origin> child element."""
    origin = element.find("origin")
    if origin is None:
        return np.eye(3), np.zeros(3)
    xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()])
    rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
    return _rpy_matrix(rpy), xyz


def link_world_transforms(urdf: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World (R, t) of every link at the zero pose.

    With all joint angles zero only the fixed <origin> transforms
    compose, so this is plain accumulation over the joint tree. The
    result is in the URDF root frame. base_link.stl, the wrist motors,
    and the wrist adapter are in their URDF link frames; the other arm STLs
    are still assembled CAD-world and are shifted by CAD_TO_BASE when they
    are attached as geoms.
    """
    root = ET.parse(urdf).getroot()
    joints = []
    children = set()
    for joint in root.iter("joint"):
        rotation, translation = _origin(joint)
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        joints.append((parent, child, rotation, translation))
        children.add(child)
    links = {link.get("name") for link in root.iter("link")}
    world = {name: (np.eye(3), np.zeros(3)) for name in links - children}
    pending = joints
    while pending:
        rest = []
        for parent, child, rotation, translation in pending:
            if parent in world:
                parent_r, parent_t = world[parent]
                world[child] = (parent_r @ rotation,
                                parent_t + parent_r @ translation)
            else:
                rest.append((parent, child, rotation, translation))
        if len(rest) == len(pending):
            raise RuntimeError(
                f"URDF joint tree disconnected at {[j[1] for j in rest]}")
        pending = rest
    return world


def link_mesh_visuals(urdf: Path, link_name: str):
    """[(mesh reference, R, t)] for each mesh visual of `link_name`,
    with (R, t) the visual origin in the link's own frame."""
    out = []
    for link in ET.parse(urdf).getroot().iter("link"):
        if link.get("name") != link_name:
            continue
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            rotation, translation = _origin(visual)
            out.append((mesh.get("filename"), rotation, translation))
    return out


def hand_rigid_visuals(urdf: Path):
    """[(mesh reference, R, t)] for the hand link and every link welded to
    it through fixed joints (e.g. the D405 on camera_frame), with (R, t)
    the visual origin in the hand link's frame."""
    fixed_children: dict[str, list] = {}
    for joint in ET.parse(urdf).getroot().iter("joint"):
        if joint.get("type") != "fixed":
            continue
        rotation, translation = _origin(joint)
        fixed_children.setdefault(joint.find("parent").get("link"), []).append(
            (joint.find("child").get("link"), rotation, translation))
    out = []
    pending = [(HAND_LINK, np.eye(3), np.zeros(3))]
    while pending:
        link, link_r, link_t = pending.pop(0)
        for reference, visual_r, visual_t in link_mesh_visuals(urdf, link):
            out.append((reference, link_r @ visual_r, link_t + link_r @ visual_t))
        for child, rotation, translation in fixed_children.get(link, []):
            pending.append((child, link_r @ rotation, link_t + link_r @ translation))
    return out


def hand_finger_joints(urdf: Path):
    """Prismatic finger joints branching off the hand link.

    [(joint name, child link, R_origin, t_origin, axis, upper, visuals)]
    with the joint origin in the hand link's frame and visuals the child
    link's [(mesh, R, t)] in its own frame. Sorted by joint name so qpos
    order is deterministic. Empty for URDFs with welded fingers.
    """
    out = []
    for joint in ET.parse(urdf).getroot().iter("joint"):
        if joint.get("type") != "prismatic":
            continue
        parent = joint.find("parent")
        if parent is None or parent.get("link") != HAND_LINK:
            continue
        child = joint.find("child").get("link")
        rotation, translation = _origin(joint)
        axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
        limit = joint.find("limit")
        upper = float(limit.get("upper")) if limit is not None else 0.0
        out.append((joint.get("name"), child, rotation, translation, axis,
                    upper, link_mesh_visuals(urdf, child)))
    out.sort(key=lambda f: f[0])
    return out


def base_stack_shift(urdf: Path,
                     world: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Where the URDF puts the shoulder bracket relative to the vendor CAD:
    its link origin plus its visual origin, against the CAD origin in the
    base_link frame (-CAD_TO_BASE). Zero for the vendor stack; it must be a
    shift along the base yaw axis, which every later link inherits."""
    visuals = [v for v in link_mesh_visuals(urdf, SHOULDER_LINK)
               if Path(v[0]).name == SHOULDER_MESH]
    if not visuals:
        raise RuntimeError(f"URDF link {SHOULDER_LINK!r} has no {SHOULDER_MESH} visual")
    _, link_t = world[SHOULDER_LINK]
    _, _, visual_t = visuals[0]
    shift = link_t + visual_t + CAD_TO_BASE
    if np.linalg.norm(shift[:2]) > 0.0005 or abs(shift[2]) > 0.1:
        raise RuntimeError(
            f"URDF moves {SHOULDER_LINK} off the base axis or implausibly far "
            f"from the CAD: {np.round(shift, 4)} m")
    return np.array([0.0, 0.0, shift[2]])


def stl_bbox_center(path: Path) -> np.ndarray:
    """Bounding-box centre of a binary STL, in metres (CAD world frame)."""
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    records = np.frombuffer(raw, dtype=np.uint8, offset=84)[: count * 50]
    tris = records.reshape(count, 50)[:, 12:48].copy().view("<f4")
    verts = tris.reshape(-1, 3) * MESH_SCALE
    return (verts.min(axis=0) + verts.max(axis=0)) / 2.0


def mesh_directory(urdf: Path) -> Path:
    """Mesh folder resolved from the URDF's own <mesh filename> entries
    (relative to the URDF directory, the controller's convention)."""
    for mesh in ET.parse(urdf).getroot().iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            candidate = (urdf.parent / filename).resolve().parent
            # The Hand-E drop ships its own base_link.stl; require an arm
            # motor housing too so its folder is never mistaken for the arm's.
            if (candidate / BASE_MESH).is_file() and (candidate / CHAIN[0][1]).is_file():
                return candidate
    raise FileNotFoundError(
        f"could not resolve a mesh directory containing {BASE_MESH} from "
        f"the mesh references in {urdf}"
    )


def urdf_axes(urdf: Path) -> dict[str, np.ndarray]:
    """Joint name -> unit axis direction, straight from the vendor URDF."""
    axes: dict[str, np.ndarray] = {}
    for joint in ET.parse(urdf).getroot().iter("joint"):
        axis = joint.find("axis")
        if axis is None:
            continue
        vec = np.array([float(v) for v in axis.get("xyz").split()])
        axes[joint.get("name")] = vec / np.linalg.norm(vec)
    return axes


def build_spec(mesh_dir: Path, axes: dict[str, np.ndarray],
               hand_pose: tuple[np.ndarray, np.ndarray],
               hand_visuals: list,
               target_pose: tuple[np.ndarray, np.ndarray],
               link_frame_t: dict[str, np.ndarray],
               finger_joints: list | None = None,
               stack: np.ndarray | None = None) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "owl69"
    spec.compiler.meshdir = "assets"
    spec.compiler.degree = False  # joint ranges below are radians
    # Dynamics settings for the hardware emulator (harmless for the
    # kinematic viewer, which only runs forward kinematics).
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    centers = {
        mesh: stl_bbox_center(mesh_dir / mesh)
        for _, motor, meshes in CHAIN
        for mesh in {motor, *meshes}
        if mesh not in LINK_FRAME_MESHES
    }
    centers[BASE_MESH] = stl_bbox_center(mesh_dir / BASE_MESH)

    # Arm STLs except the baked link-frame housings are still CAD-world:
    # CAD vertex - cad_shift(mesh) is the vertex in the base_link frame.
    stack = np.zeros(3) if stack is None else np.asarray(stack, dtype=float)

    def cad_shift(mesh: str) -> np.ndarray:
        if mesh in (BASE_MESH, BASE_ACTUATOR_MESH):
            return CAD_TO_BASE
        return CAD_TO_BASE - stack

    centers = {mesh: c - cad_shift(mesh) for mesh, c in centers.items()}
    for mesh, t in link_frame_t.items():
        centers[mesh] = np.asarray(t, dtype=float)

    mesh_assets: dict[str, str] = {}

    def add_mesh_asset(name: str) -> str:
        if name not in mesh_assets:
            asset = spec.add_mesh()
            asset.name = Path(name).stem
            asset.file = name
            asset.scale = [MESH_SCALE] * 3
            asset.maxhullvert = 64  # collision hulls stay cheap on the Quest
            mesh_assets[name] = asset.name
        return mesh_assets[name]

    def add_visual(body, mesh: str, body_world: np.ndarray,
                   *, vertex_origin: np.ndarray | None = None) -> None:
        # CAD-world meshes render at (CAD - cad_shift). Baked meshes
        # (base_link.stl, the wrist motors, the wrist adapter) have
        # vertices in a URDF link frame whose world origin is `vertex_origin`.
        if vertex_origin is None:
            pos = -(body_world + cad_shift(mesh))
        else:
            pos = vertex_origin - body_world
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = add_mesh_asset(mesh)
        geom.pos = pos
        geom.rgba = SILVER
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geom.density = 1000.0
        # Collision twin: convex hull of the same mesh, invisible to the
        # headset renderer (group 3, contype set). Adjacent links are
        # parent-child pairs, which MuJoCo excludes from contact itself.
        col = body.add_geom()
        col.type = mujoco.mjtGeom.mjGEOM_MESH
        col.meshname = mesh_assets[mesh]
        col.pos = pos
        col.group = 3
        col.density = 0.0

    world = spec.worldbody
    # Solid floor so the gripper stops at the ground. It lives on the world
    # body, as do the base plate's geoms: same-body pairs never collide,
    # and link1 is the world's child (parent-child pairs are excluded too),
    # so only link2 onward can actually hit the floor.
    floor = world.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2.0, 2.0, 0.1]
    floor.group = 3
    add_visual(world, BASE_MESH, np.zeros(3), vertex_origin=np.zeros(3))

    parent = world
    parent_origin = np.zeros(3)
    body_names = []
    for index, (joint_name, motor_mesh, moving_meshes) in enumerate(CHAIN):
        origin = centers[motor_mesh]
        body = parent.add_body()
        body.name = (
            EE_BODY if index == len(CHAIN) - 1 else f"link{index + 1}"
        )
        body.pos = origin - parent_origin
        body_names.append(body.name)

        joint = body.add_joint()
        joint.name = joint_name
        joint.type = mujoco.mjtJoint.mjJNT_HINGE
        joint.axis = axes[joint_name]
        _, effort, kp, kd, damping = JOINT_DRIVES[joint_name]
        joint.range = [-JOINT_RANGE, JOINT_RANGE]
        # damping is per-DOF in the spec API; a hinge uses element 0.
        joint.damping = [damping, 0.0, 0.0]
        joint.armature = 0.05

        # Position servo (PD in actuator space) for the hardware emulator.
        # The kinematic teleop path never writes ctrl, so it is unaffected.
        act = spec.add_actuator()
        act.name = f"servo_{joint_name}"
        act.target = joint_name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        # Whole-array assignment: element writes land on a property copy.
        gainprm = np.zeros(mujoco.mjNGAIN)
        gainprm[0] = kp
        act.gainprm = gainprm
        biasprm = np.zeros(mujoco.mjNBIAS)
        biasprm[1] = -kp
        biasprm[2] = -kd
        act.biasprm = biasprm
        act.forcerange = [-effort, effort]
        act.ctrlrange = [-JOINT_RANGE, JOINT_RANGE]

        for mesh in moving_meshes:
            if mesh in LINK_FRAME_MESHES:
                add_visual(body, mesh, origin, vertex_origin=link_frame_t[mesh])
            else:
                add_visual(body, mesh, origin)

        if index == len(CHAIN) - 1:
            # tcp site: the URDF's target_frame (the controller tip),
            # position and orientation.
            site = body.add_site()
            site.name = TCP_SITE
            target_r, target_t = target_pose
            site.pos = target_t - origin
            site_quat = np.zeros(4)
            mujoco.mju_mat2Quat(site_quat,
                                np.ascontiguousarray(target_r.flatten()))
            site.quat = site_quat
            site.size = [0.005, 0.005, 0.005]

        parent = body
        parent_origin = origin

    # -- Robotiq Hand-E, placed exactly where the URDF puts it --
    # The URDF's hand_e_mount fixed joint (gripper flange on the EE flange
    # face, rpy [0, 0, -pi/2]) composed by zero-pose forward kinematics is the
    # single source of truth for the hand pose, and each hand mesh sits at
    # its own URDF visual origin inside that frame. Nothing is reconstructed:
    # the previous tool-axis-aligned guess left the roll about the flange
    # axis arbitrary and rendered the hand rotated 90 degrees against the
    # real robot. The gripper stays a jointless child body (a rigid weld,
    # zero degrees of freedom) so the frame is addressable by name.
    ee_body = parent
    hand_r, hand_t = hand_pose
    hand = ee_body.add_body()
    hand.name = HAND_BODY
    hand.pos = hand_t - parent_origin
    hand_quat = np.zeros(4)
    mujoco.mju_mat2Quat(hand_quat, np.ascontiguousarray(hand_r.flatten()))
    hand.quat = hand_quat
    for reference, visual_r, visual_t in hand_visuals:
        mesh = Path(reference).name
        visual_quat = np.zeros(4)
        mujoco.mju_mat2Quat(visual_quat,
                            np.ascontiguousarray(visual_r.flatten()))
        geom = hand.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = add_mesh_asset(mesh)
        geom.pos = visual_t
        geom.quat = visual_quat
        geom.rgba = SILVER
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geom.density = 1000.0
        col = hand.add_geom()
        col.type = mujoco.mjtGeom.mjGEOM_MESH
        col.meshname = mesh_assets[mesh]
        col.pos = visual_t
        col.quat = visual_quat
        col.group = 3
        col.density = 0.0

    # "target" site: the hand link frame itself. hand_e_base +X points
    # along the fingers (away from the arm); that is the add-on convention.

    target = hand.add_site()
    target.name = TARGET_SITE
    target.size = [0.005, 0.005, 0.005]

    # Articulated fingers: one slide-jointed child body of the hand per
    # URDF finger joint, posed by direct qpos writes (the hologram feeds
    # gripper feedback * stroke). No actuator: the plant's spring holds
    # the fingers at 0 (open) so free dynamics never flop them. Being the
    # deepest bodies in the tree, their qpos indices come AFTER the six
    # arm joints, so the emulator's qpos[:6] arm mapping is unaffected.
    for f_name, f_child, f_r, f_t, f_axis, f_upper, f_visuals in (
            finger_joints or []):
        finger = hand.add_body()
        finger.name = f_child
        finger.pos = f_t
        finger_quat = np.zeros(4)
        mujoco.mju_mat2Quat(finger_quat, np.ascontiguousarray(f_r.flatten()))
        finger.quat = finger_quat
        fj = finger.add_joint()
        fj.name = f_name
        fj.type = mujoco.mjtJoint.mjJNT_SLIDE
        fj.axis = f_axis
        fj.range = [0.0, f_upper]
        fj.stiffness = [5000.0, 0.0, 0.0]  # ~1 mm sag under finger weight
        fj.damping = [10.0, 0.0, 0.0]
        for reference, visual_r, visual_t in f_visuals:
            mesh = Path(reference).name
            visual_quat = np.zeros(4)
            mujoco.mju_mat2Quat(visual_quat,
                                np.ascontiguousarray(visual_r.flatten()))
            geom = finger.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = add_mesh_asset(mesh)
            geom.pos = visual_t
            geom.quat = visual_quat
            geom.rgba = SILVER
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 2
            geom.density = 1000.0
            col = finger.add_geom()
            col.type = mujoco.mjtGeom.mjGEOM_MESH
            col.meshname = mesh_assets[mesh]
            col.pos = visual_t
            col.quat = visual_quat
            col.group = 3
            col.density = 0.0

    return spec


def validate(model: mujoco.MjModel) -> None:
    """A wrong anchor makes the joint's own motor housing orbit the axis."""
    data = mujoco.MjData(model)
    mesh_of_geom: dict[int, str] = {}
    for g in range(model.ngeom):
        mesh_id = model.geom_dataid[g]
        if mesh_id >= 0:
            mesh_of_geom[g] = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
            )

    def center_of(mesh_stem: str) -> np.ndarray:
        for g, stem in mesh_of_geom.items():
            if stem == mesh_stem:
                return data.geom_xpos[g].copy()
        raise KeyError(mesh_stem)

    for j, (joint_name, motor_mesh, _) in enumerate(CHAIN):
        stem = Path(motor_mesh).stem
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        before = center_of(stem)
        data.qpos[j] = 0.5
        mujoco.mj_forward(model, data)
        drift = float(np.linalg.norm(center_of(stem) - before))
        if drift > 0.03:
            raise RuntimeError(
                f"{joint_name}: motor housing moved {drift:.3f} m during a "
                "joint sweep; the anchor is off its own axis"
            )


def _build_bundle(urdf: Path, fingerprint: str, force: bool) -> Path:
    """Run the recipe against `urdf` unless the stamp already matches."""
    stamp = BUNDLE_DIR / ".source"
    if (
        not force
        and MODEL_XML.is_file()
        and stamp.is_file()
        and stamp.read_text() == fingerprint
    ):
        return MODEL_XML

    mesh_dir = mesh_directory(urdf)

    axes = urdf_axes(urdf)
    missing = [joint for joint, _, _ in CHAIN if joint not in axes]
    if missing:
        raise RuntimeError(f"URDF is missing expected joints: {missing}")

    world = link_world_transforms(urdf)
    for link in (HAND_LINK, TARGET_LINK, EE_BODY):
        if link not in world:
            raise RuntimeError(
                f"URDF has no {link!r} link; the hand and tcp placement "
                "come from the URDF, so an export without them cannot be "
                "converted")
    hand_visuals = hand_rigid_visuals(urdf)
    if not hand_visuals:
        raise RuntimeError(f"URDF link {HAND_LINK!r} carries no mesh visuals")

    # Frame consistency: the EE STL is still assembled CAD-world, so
    # its visual origin plus the URDF link origin must equal the CAD
    # origin expressed in the baked base_link frame (-CAD_TO_BASE), moved
    # by the base-stack correction it inherits from the shoulder bracket.
    stack = base_stack_shift(urdf, world)
    flange_visuals = link_mesh_visuals(urdf, EE_BODY)
    if flange_visuals:
        _, flange_t = world[EE_BODY]
        _, _, visual_t = flange_visuals[0]
        cancel = np.linalg.norm(flange_t + visual_t + CAD_TO_BASE - stack)
        if cancel > 0.005:
            raise RuntimeError(
                f"URDF frame convention changed: {EE_BODY} visual origin "
                f"does not cancel its link origin (residual {cancel:.4f} m)")

    # Rebuilt from scratch: the teleop host ships every STL in assets/ to
    # the headset, so meshes of a previous model must not linger.
    assets = BUNDLE_DIR / "assets"
    shutil.rmtree(assets, ignore_errors=True)
    assets.mkdir(parents=True)
    needed = {BASE_MESH} | {
        mesh for _, motor, meshes in CHAIN for mesh in {motor, *meshes}
    }
    for name in sorted(needed):
        source = mesh_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"mesh missing from vendor drop: {source}")
        shutil.copy2(source, assets / name)
    finger_joints = hand_finger_joints(urdf)
    finger_visuals = [v for f in finger_joints for v in f[6]]
    # assets/ is flat and the headset VFS resolves bare names, so a hand
    # mesh sharing a file name with another mesh (the Hand-E drop's own
    # base_link.stl) is stored under its folder-prefixed name.
    asset_name: dict[str, str] = {}
    for reference, _, _ in hand_visuals + finger_visuals:
        if reference in asset_name:
            continue
        source = (urdf.parent / reference).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"hand mesh missing: {source}")
        name = source.name
        if name in needed or name in asset_name.values():
            name = f"{source.parent.name}_{name}"
        asset_name[reference] = name
        shutil.copy2(source, assets / name)
    hand_visuals = [(asset_name[ref], r, t) for ref, r, t in hand_visuals]
    finger_joints = [
        (*f[:6], [(asset_name[ref], r, t) for ref, r, t in f[6]])
        for f in finger_joints
    ]

    spec = build_spec(mesh_dir, axes, world[HAND_LINK], hand_visuals,
                      world[TARGET_LINK], {
                          J5_MESH: world["AD_50070_J4_v2_2"][1],
                          J6_MESH: world["AD_50070_J4_v2_3"][1],
                          ADAPTER_MESH: world["wrist_adapter_link"][1],
                      }, finger_joints=finger_joints, stack=stack)
    # Compile against the absolute assets dir (an in-memory spec has no
    # model directory; to_xml() recompiles too), then swap in the portable
    # relative path for the saved bundle.
    spec.compiler.meshdir = str(assets)
    model = spec.compile()  # raises on structural problems
    validate(model)
    xml = spec.to_xml().replace(f'meshdir="{assets}"', 'meshdir="assets"')
    MODEL_XML.write_text(xml)

    # The bundle must stand on its own (this is what the headset compiles).
    bundled = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    validate(bundled)
    for site in (TCP_SITE, TARGET_SITE):
        if mujoco.mj_name2id(bundled, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
            raise RuntimeError(f"{site} site missing from the bundled model")

    stamp.write_text(fingerprint)
    return MODEL_XML


def prepare(urdf: Path | None = None, force: bool = False) -> Path:
    """Build (or reuse) the MJCF bundle from a local URDF path."""
    urdf = vendor_urdf() if urdf is None else urdf.expanduser().resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    fingerprint = f"v12:{urdf}:{urdf.stat().st_mtime_ns}"  # v12: base-stack shift
    return _build_bundle(urdf, fingerprint, force)


def prepare_from_controller(endpoint: str, force: bool = False,
                            served: tuple[Path, str] | None = None) -> Path:
    """Build (or reuse) the bundle from the model a running controller
    serves over ZMQ (GET_MODEL for the URDF, GET_ASSET per mesh).

    The controller owns the model: the fetched URDF and meshes are
    mirrored under owl69_mjcf/served/ and the fingerprint is a content
    hash, so the bundle rebuilds exactly when the served model changes.
    ``served`` is a fetch_served_model() result to build from instead of
    fetching again. Raises on any transport or protocol failure (callers
    retry).
    """
    urdf_path, fingerprint = served or fetch_served_model(endpoint)
    return _build_bundle(urdf_path, fingerprint, force)


def fetch_served_model(endpoint: str) -> tuple[Path, str]:
    """Mirror the URDF and meshes a running controller serves (GET_MODEL,
    GET_ASSET per mesh) under owl69_mjcf/served/; returns the mirrored
    URDF path and a content fingerprint of the whole model. Raises on
    any transport or protocol failure."""
    import zmq

    import robot_interface as ri

    req = ri.connect_req(endpoint)
    try:
        req.send(ri.encode_get_model())
        model = ri.parse_model(req.recv())
        if model is None:
            raise RuntimeError("controller did not answer GET_MODEL")
        urdf_bytes = model["urdf_xml"]

        refs = sorted({
            mesh.get("filename")
            for mesh in ET.fromstring(urdf_bytes).iter("mesh")
            if mesh.get("filename")
        })
        if not refs:
            raise RuntimeError("served URDF references no meshes")

        urdf_path = SERVED_DIR / "robot" / "model.urdf"
        digest = hashlib.sha256(urdf_bytes)
        assets: dict[Path, bytes] = {}
        for name in refs:
            req.send(ri.encode_get_asset(name))
            answer = req.recv()
            parsed = ri.parse_asset(answer)
            if parsed is None:
                reply = ri.parse_reply(answer)
                raise RuntimeError(f"asset {name!r} refused: "
                                   f"{reply['message'] if reply else 'bad frame'}")
            _, blob = parsed
            digest.update(name.encode())
            digest.update(hashlib.sha256(blob).digest())
            target = (urdf_path.parent / name).resolve()
            if SERVED_DIR.resolve() not in target.parents:
                raise RuntimeError(f"asset path escapes served/: {name!r}")
            assets[target] = blob
    except zmq.error.Again as exc:
        raise RuntimeError(f"controller at {endpoint} not answering") from exc
    finally:
        req.close(0)

    for target, blob in assets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_path.write_bytes(urdf_bytes)

    total = sum(len(blob) for blob in assets.values())
    print(f"served model fetched: {len(urdf_bytes)} B URDF, "
          f"{len(assets)} assets ({total / 1e6:.1f} MB)")
    return urdf_path, f"w4:{digest.hexdigest()}"  # w4: welded hand links


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--from-controller", metavar="ENDPOINT", default=None,
                        help="fetch the URDF and meshes from a running "
                             "controller's command socket (e.g. "
                             "tcp://127.0.0.1:5556) instead of local paths")
    parser.add_argument("--force", action="store_true", help="rebuild even if fresh")
    args = parser.parse_args()
    if args.from_controller and args.urdf:
        parser.error("--from-controller and --urdf are mutually exclusive")
    if args.from_controller:
        path = prepare_from_controller(args.from_controller, force=args.force)
    else:
        path = prepare(args.urdf, force=args.force)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = data.site_xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    ]
    print(f"bundle ready: {path}")
    print(f"  nq={model.nq} nbody={model.nbody} ngeom={model.ngeom}")
    print(f"  tcp at zero pose: {np.round(site, 3)}")


if __name__ == "__main__":
    main()
