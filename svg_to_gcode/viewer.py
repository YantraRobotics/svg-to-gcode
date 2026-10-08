import math
import time
import xml.etree.ElementTree as ET
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import zmq

from robot_interface import Shell
from svg_to_gcode.robot import MM_PER_M, PEN_STATUS_ENDPOINT, Placement

type Texel = tuple[float, float]
type Pose = tuple[np.ndarray, np.ndarray]
type Rgb = tuple[int, int, int]

PAPER = "paper"
PEN = "pen"
PEN_TIP = "pen_tip"
TCP_SITE = "tcp"
ARM_JOINTS = 6
TEXELS_PER_MM = 4.0
INK_RADIUS_TEXELS = 1
# Ink before the first pen announcement, and for G-code without pen changes.
INK_RGB = (20, 30, 90)
CHANNEL_MAX = 255
PAPER_RGB = (0.97, 0.97, 0.95)
# A pen tip this close above the sheet, or anywhere below it, leaves ink.
INK_CONTACT_MM = 0.5
PEN_RADIUS_M = 0.005
PEN_GRIP_M = 0.06
PEN_RGBA = (0.15, 0.35, 0.8, 1.0)
DESK_MARGIN_M = 0.04
DESK_GAP_M = 0.0005
DESK_RGBA = (0.55, 0.4, 0.25, 1.0)
PLANE_GRID_M = 0.01
# prepare_owl69_model puts the robot's rendered meshes in geom group 2.
VISUAL_GROUP = 2
UNCOLORED_RGBA = (0.7, 0.7, 0.7, 1.0)
SKY = "sky"
SKY_TOP_RGB = (0.3, 0.5, 0.7)
SKY_BOTTOM_RGB = (0.0, 0.0, 0.0)
SKY_TEXELS = 512
FLOOR = "floor"
FLOOR_RGB = ((0.2, 0.3, 0.4), (0.1, 0.2, 0.3))
FLOOR_MARK_RGB = (0.8, 0.8, 0.8)
FLOOR_TEXELS = 300
FLOOR_REPEATS_PER_M = 5.0
FLOOR_REFLECTANCE = 0.2
FLOOR_HALF_SIZE_M = 1.5
LIGHT_HEIGHT_M = 1.5
HEADLIGHT_AMBIENT = (0.3, 0.3, 0.3)
HEADLIGHT_DIFFUSE = (0.6, 0.6, 0.6)
HEADLIGHT_SPECULAR = (0.0, 0.0, 0.0)
FRAME_PERIOD_S = 1.0 / 60.0
TELEMETRY_TIMEOUT_S = 0.1
CAMERA_DISTANCE_M = 1.3
CAMERA_ELEVATION_DEG = -35.0
CAMERA_HEIGHT_M = 0.15


@dataclass(frozen=True)
class Visual:
    """A URDF <visual> mesh with its colour and its pose in the link frame."""

    link: str
    mesh: Path
    scale: tuple[float, ...]
    rgba: tuple[float, ...]
    pose: Pose


def vector(text: str, size: int, what: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split())
    if len(values) != size:
        raise ValueError(f"URDF {what} needs {size} numbers, got {text!r}")
    return values


def urdf_visuals(urdf: Path) -> list[Visual]:
    root = ET.parse(urdf).getroot()
    colors = {
        material.get("name"): vector(color.get("rgba"), 4, "rgba")
        for material in root.findall("material")
        if (color := material.find("color")) is not None
    }
    visuals = []
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            rgba = UNCOLORED_RGBA
            if (material := visual.find("material")) is not None:
                color = material.find("color")
                rgba = (
                    vector(color.get("rgba"), 4, "rgba")
                    if color is not None
                    else colors.get(material.get("name"), UNCOLORED_RGBA)
                )
            origin = visual.find("origin")
            xyz = "0 0 0" if origin is None else origin.get("xyz", "0 0 0")
            rpy = "0 0 0" if origin is None else origin.get("rpy", "0 0 0")
            quat = np.zeros(4)
            # URDF rpy turns about the fixed X, then Y, then Z axes.
            mujoco.mju_euler2Quat(quat, np.array(vector(rpy, 3, "rpy")), "XYZ")
            visuals.append(
                Visual(
                    link=link.get("name"),
                    mesh=(urdf.parent / mesh.get("filename")).resolve(),
                    scale=vector(mesh.get("scale", "1 1 1"), 3, "mesh scale"),
                    rgba=rgba,
                    pose=(np.array(vector(xyz, 3, "xyz")), quat),
                )
            )
    return visuals


def compose(first: Pose, second: Pose) -> Pose:
    pos, quat = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(pos, quat, first[0], first[1], second[0], second[1])
    return pos, quat


def invert(pose: Pose) -> Pose:
    pos, quat = np.zeros(3), np.zeros(4)
    mujoco.mju_negPose(pos, quat, pose[0], pose[1])
    return pos, quat


def bundle_mesh(mesh_names: dict[str, str], visual: Visual) -> str | None:
    """The bundle's mesh asset for a URDF visual, or None when the bundle leaves it out.

    The bundle keeps every mesh in one folder and prefixes a file with its URDF
    folder name when another mesh already took the plain name.
    """
    for file in (f"{visual.mesh.parent.name}_{visual.mesh.name}", visual.mesh.name):
        if file in mesh_names:
            return mesh_names[file]
    return None


def dress_robot(spec: mujoco.MjSpec, urdf: Path) -> None:
    """Paint the bundle's meshes in their URDF colours and add the visuals it leaves out.

    A missing visual (the printed covers) is placed relative to a mesh of the same
    URDF link that the bundle has, so it inherits the bundle's fixes to the vendor frames.
    """
    mesh_names = {mesh.file: mesh.name for mesh in spec.meshes}
    geoms_by_mesh: dict[str, list[mujoco.MjsGeom]] = {}
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.group == VISUAL_GROUP:
            geoms_by_mesh.setdefault(geom.meshname, []).append(geom)

    anchors: dict[str, tuple[Visual, mujoco.MjsGeom]] = {}
    missing = []
    for visual in urdf_visuals(urdf):
        geoms = geoms_by_mesh.get(bundle_mesh(mesh_names, visual), [])
        if not geoms:
            missing.append(visual)
            continue
        for geom in geoms:
            geom.rgba = visual.rgba
        if len(geoms) == 1:
            anchors.setdefault(visual.link, (visual, geoms[0]))

    added: dict[Path, str] = {}
    for visual in missing:
        if visual.link not in anchors:
            raise ValueError(
                f"cannot place {visual.mesh.name}: URDF link {visual.link} "
                "has no mesh in the MuJoCo bundle"
            )
        if not visual.mesh.is_file():
            raise FileNotFoundError(f"URDF mesh not found: {visual.mesh}")
        anchor, anchor_geom = anchors[visual.link]
        link_in_body = compose((anchor_geom.pos, anchor_geom.quat), invert(anchor.pose))
        pos, quat = compose(link_in_body, visual.pose)
        if visual.mesh not in added:
            mesh = spec.add_mesh(
                name=f"urdf_{visual.mesh.parent.name}_{visual.mesh.stem}",
                file=str(visual.mesh),
                scale=visual.scale,
                # Printed covers are thin shells, not closed volumes.
                inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
            )
            added[visual.mesh] = mesh.name
        anchor_geom.parent.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=added[visual.mesh],
            pos=pos,
            quat=quat,
            rgba=visual.rgba,
            group=VISUAL_GROUP,
            contype=0,
            conaffinity=0,
            density=0.0,
        )


def add_textured_material(spec: mujoco.MjSpec, name: str, **settings) -> None:
    material = spec.add_material(name=name, **settings)
    textures = list(material.textures)
    textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    material.textures = textures


def add_surroundings(spec: mujoco.MjSpec, floor_z: float) -> None:
    """A gradient sky, a checkered floor at floor_z, and an overhead light."""
    spec.add_texture(
        name=SKY,
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=SKY_TOP_RGB,
        rgb2=SKY_BOTTOM_RGB,
        width=SKY_TEXELS,
    )
    spec.add_texture(
        name=FLOOR,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=FLOOR_RGB[0],
        rgb2=FLOOR_RGB[1],
        markrgb=FLOOR_MARK_RGB,
        width=FLOOR_TEXELS,
        height=FLOOR_TEXELS,
    )
    add_textured_material(
        spec,
        FLOOR,
        texrepeat=[FLOOR_REPEATS_PER_M, FLOOR_REPEATS_PER_M],
        texuniform=True,
        reflectance=FLOOR_REFLECTANCE,
    )
    spec.worldbody.add_geom(
        name=FLOOR,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[FLOOR_HALF_SIZE_M, FLOOR_HALF_SIZE_M, PLANE_GRID_M],
        pos=[0.0, 0.0, floor_z],
        material=FLOOR,
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.0, 0.0, floor_z + LIGHT_HEIGHT_M],
        dir=[0.0, 0.0, -1.0],
        castshadow=True,
    )
    spec.visual.headlight.ambient = HEADLIGHT_AMBIENT
    spec.visual.headlight.diffuse = HEADLIGHT_DIFFUSE
    spec.visual.headlight.specular = HEADLIGHT_SPECULAR


def build_scene(
    bundle: Path,
    urdf: Path,
    paper: tuple[float, float],
    placement: Placement,
    pen_offset: float,
) -> mujoco.MjModel:
    """The robot bundle in its URDF look, on a floor, with a desk, an inkable sheet, and the gripped pen."""
    spec = mujoco.MjSpec.from_file(str(bundle))
    dress_robot(spec, urdf)
    width, height = paper
    center = placement.to_base((width / 2, height / 2, 0.0))
    # The floor is level with the robot's base unless the paper lies lower.
    desk_top = center[2] - DESK_GAP_M
    floor_z = min(0.0, desk_top)
    add_surroundings(spec, floor_z)

    spec.add_texture(
        name=PAPER,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=PAPER_RGB,
        width=round(width * TEXELS_PER_MM),
        height=round(height * TEXELS_PER_MM),
    )
    add_textured_material(spec, PAPER)
    yaw = [math.cos(placement.yaw / 2), 0.0, 0.0, math.sin(placement.yaw / 2)]
    half_width, half_height = width / 2 / MM_PER_M, height / 2 / MM_PER_M
    spec.worldbody.add_geom(
        name=PAPER,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[half_width, half_height, PLANE_GRID_M],
        pos=center,
        quat=yaw,
        material=PAPER,
        contype=0,
        conaffinity=0,
    )
    desk_half_height = (desk_top - floor_z) / 2
    if desk_half_height > 0:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                half_width + DESK_MARGIN_M,
                half_height + DESK_MARGIN_M,
                desk_half_height,
            ],
            pos=[center[0], center[1], floor_z + desk_half_height],
            quat=yaw,
            rgba=DESK_RGBA,
            contype=0,
            conaffinity=0,
        )

    tcp = spec.site(TCP_SITE)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, tcp.quat)
    axis = rotation.reshape(3, 3)[:, 0]
    tip = tcp.pos + pen_offset / MM_PER_M * axis
    tcp.parent.add_geom(
        name=PEN,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[PEN_RADIUS_M, 0.0, 0.0],
        fromto=[*(tcp.pos - PEN_GRIP_M * axis), *tip],
        rgba=PEN_RGBA,
        contype=0,
        conaffinity=0,
    )
    tcp.parent.add_site(name=PEN_TIP, pos=tip)
    return spec.compile()


def paper_texels(model: mujoco.MjModel) -> np.ndarray:
    """Writable rows x columns x RGB view of the paper texture; row 0 is the top edge."""
    texture = model.texture(PAPER).id
    start = model.tex_adr[texture]
    shape = (
        model.tex_height[texture],
        model.tex_width[texture],
        model.tex_nchannel[texture],
    )
    return model.tex_data[start : start + math.prod(shape)].reshape(shape)


def to_texel(x: float, y: float, paper_height: float) -> Texel:
    """Paper mm (origin bottom-left, Y up) to texture (row, column)."""
    return (paper_height - y) * TEXELS_PER_MM, x * TEXELS_PER_MM


def ink(texels: np.ndarray, start: Texel, end: Texel, rgb: Rgb) -> None:
    samples = math.ceil(max(abs(end[0] - start[0]), abs(end[1] - start[1]))) + 1
    rows = np.rint(np.linspace(start[0], end[0], samples)).astype(int)
    columns = np.rint(np.linspace(start[1], end[1], samples)).astype(int)
    last_row, last_column = texels.shape[0] - 1, texels.shape[1] - 1
    for row_step in range(-INK_RADIUS_TEXELS, INK_RADIUS_TEXELS + 1):
        for column_step in range(-INK_RADIUS_TEXELS, INK_RADIUS_TEXELS + 1):
            texels[
                np.clip(rows + row_step, 0, last_row),
                np.clip(columns + column_step, 0, last_column),
            ] = rgb


def subscribe_pen_status(endpoint: str = PEN_STATUS_ENDPOINT) -> zmq.Socket:
    """A SUB socket for draw's pen announcements that keeps only the newest one."""
    socket = zmq.Context.instance().socket(zmq.SUB)
    # CONFLATE only applies to later connects; LINGER 0 keeps closing from
    # waiting on a subscription that never reached a publisher.
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.subscribe(b"")
    socket.connect(endpoint)
    return socket


def latest_pen_rgb(socket: zmq.Socket) -> Rgb | None:
    """The newest announced pen colour, or None when no announcement is waiting."""
    # Polling a SUB socket prefetches a message that CONFLATE cannot replace,
    # so drain everything waiting and keep the last.
    messages = []
    with suppress(zmq.Again):
        while True:
            messages.append(socket.recv_json(zmq.NOBLOCK))
    if not messages:
        return None
    status = messages[-1]
    rgb = status.get("rgb") if isinstance(status, dict) else None
    valid = (
        isinstance(rgb, list)
        and len(rgb) == 3
        and all(isinstance(channel, int) for channel in rgb)
        and all(0 <= channel <= CHANNEL_MAX for channel in rgb)
    )
    if not valid:
        raise ValueError(
            f"pen status needs rgb as three integers 0..{CHANNEL_MAX}, got {status!r}"
        )
    return tuple(rgb)


def pen_rgba(rgb: Rgb) -> tuple[float, float, float, float]:
    return (*(channel / CHANNEL_MAX for channel in rgb), 1.0)


def frame_camera(
    camera: mujoco.MjvCamera, placement: Placement, paper: tuple[float, float]
) -> None:
    """Look from the side at the midpoint between the robot base and the paper."""
    center = placement.to_base((paper[0] / 2, paper[1] / 2, 0.0))
    camera.lookat[:] = (center[0] / 2, center[1] / 2, center[2] + CAMERA_HEIGHT_M)
    camera.distance = CAMERA_DISTANCE_M
    camera.azimuth = math.degrees(math.atan2(center[1], center[0])) + 90.0
    camera.elevation = CAMERA_ELEVATION_DEG


def show(
    shell: Shell,
    model: mujoco.MjModel,
    placement: Placement,
    paper: tuple[float, float],
) -> None:
    """Mirror the arm from telemetry and ink the paper wherever the pen tip touches it.

    The ink and the pen take the colour of the pen that draw last announced.
    """
    width, height = paper
    data = mujoco.MjData(model)
    texture = model.texture(PAPER).id
    texels = paper_texels(model)
    tip_site = model.site(PEN_TIP).id
    pen_geom = model.geom(PEN).id
    ink_rgb = INK_RGB
    previous = None
    with (
        subscribe_pen_status() as pen_status,
        mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer,
    ):
        with viewer.lock():
            frame_camera(viewer.cam, placement, paper)
        while viewer.is_running():
            started = time.monotonic()
            announced = latest_pen_rgb(pen_status)
            if announced is not None:
                ink_rgb = announced
                with viewer.lock():
                    model.geom_rgba[pen_geom] = pen_rgba(announced)
            telemetry = shell.telemetry(TELEMETRY_TIMEOUT_S)
            if telemetry is not None:
                with viewer.lock():
                    data.qpos[:ARM_JOINTS] = telemetry["q"]
                    mujoco.mj_forward(model, data)
                    x, y, z = placement.to_paper(data.site_xpos[tip_site])
                    on_sheet = 0.0 <= x <= width and 0.0 <= y <= height
                    current = None
                    if on_sheet and z <= INK_CONTACT_MM:
                        current = to_texel(x, y, height)
                        ink(texels, previous or current, current, ink_rgb)
                previous = current
                if current is not None:
                    viewer.update_texture(texture)
                viewer.sync()
            time.sleep(max(0.0, FRAME_PERIOD_S - (time.monotonic() - started)))
