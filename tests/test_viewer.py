import math
import struct
import time
from contextlib import closing
from pathlib import Path

import mujoco
import numpy as np
import pytest
import zmq

from svg_to_gcode import viewer
from svg_to_gcode.pens import Pen
from svg_to_gcode.robot import PenAnnouncer, Placement

PAPER_MM = (80.0, 80.0)
PEN_OFFSET_MM = 40.0
CYAN = Pen("cyan", (0, 174, 239))
MAGENTA = Pen("magenta", (236, 0, 140))
RECEIVE_TIMEOUT_S = 2.0
POLL_MS = 10
BUNDLE = """
<mujoco>
  <worldbody>
    <body name="end_effector_link" pos="0.3 0 0.5">
      <site name="tcp" quat="0.7071068 0 0.7071068 0"/>
    </body>
  </worldbody>
</mujoco>
"""
EMPTY_URDF = '<robot name="empty"/>'
DRESSED_BUNDLE = """
<mujoco>
  <compiler meshdir="assets"/>
  <asset>
    <mesh name="anchor" file="anchor.stl" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <body name="flange" pos="0.3 0 0.5">
      <geom type="mesh" mesh="anchor" pos="0.1 0 0" group="2" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
FLANGE_LINK = """
<link name="flange">
  <visual>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <geometry><mesh filename="../assets/anchor.stl" scale="0.001 0.001 0.001"/></geometry>
    <material name="orange"/>
  </visual>
  <visual>
    <origin xyz="0 0.2 0" rpy="0 0 1.5707963267948966"/>
    <geometry><mesh filename="../covers/cover.stl" scale="0.001 0.001 0.001"/></geometry>
    <material name="black"><color rgba="0.1 0.1 0.1 1"/></material>
  </visual>
</link>
"""
ORPHAN_LINK = """
<link name="orphan">
  <visual><geometry><mesh filename="../covers/cover.stl"/></geometry></visual>
</link>
"""
# MuJoCo needs outward-wound triangles to get a positive mesh volume.
CUBE_CORNERS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]  # fmt: skip
CUBE_TRIANGLES = [
    (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
    (2, 3, 7), (2, 7, 6), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
]  # fmt: skip


def write_box_stl(path: Path, half_sizes: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [struct.pack("<80sI", b"box", len(CUBE_TRIANGLES))]
    for triangle in CUBE_TRIANGLES:
        vertices = [
            coordinate * half
            for index in triangle
            for coordinate, half in zip(CUBE_CORNERS[index], half_sizes)
        ]
        records.append(struct.pack("<12fH", 0.0, 0.0, 0.0, *vertices, 0))
    path.write_bytes(b"".join(records))


def top_down_image(model: mujoco.MjModel, center: tuple[float, ...]) -> np.ndarray:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = center
    camera.distance = 0.09
    camera.azimuth = 90.0
    camera.elevation = -90.0
    with mujoco.Renderer(model, height=200, width=200) as renderer:
        renderer.update_scene(data, camera=camera)
        return renderer.render().mean(axis=2)


@pytest.mark.parametrize("yaw_deg", [0.0, 90.0])
def test_ink_lands_where_the_pen_touches(tmp_path, yaw_deg):
    bundle = tmp_path / "bundle.xml"
    bundle.write_text(BUNDLE)
    urdf = tmp_path / "model.urdf"
    urdf.write_text(EMPTY_URDF)
    placement = Placement((260.0, -40.0, 0.0), math.radians(yaw_deg), 0.0)
    model = viewer.build_scene(bundle, urdf, PAPER_MM, placement, PEN_OFFSET_MM)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tip = data.site_xpos[model.site(viewer.PEN_TIP).id]
    assert tip == pytest.approx([0.3, 0.0, 0.5 - PEN_OFFSET_MM / 1000.0], abs=1e-6)
    assert model.geom_rgba[model.geom(viewer.PEN).id] == pytest.approx(viewer.PEN_RGBA)

    width, height = PAPER_MM
    texels = viewer.paper_texels(model)
    for y in np.arange(height / 2 + 5, height - 5, 0.25):
        start = viewer.to_texel(width / 2 + 5, y, height)
        end = viewer.to_texel(width - 5, y, height)
        viewer.ink(texels, start, end, viewer.INK_RGB)

    image = top_down_image(model, placement.to_base((width / 2, height / 2, 0.0)))
    half = image.shape[0] // 2
    quarters = {
        "top-left": image[:half, :half].mean(),
        "top-right": image[:half, half:].mean(),
        "bottom-left": image[half:, :half].mean(),
        "bottom-right": image[half:, half:].mean(),
    }
    inked = "top-right" if yaw_deg == 0.0 else "top-left"
    blank = min(value for name, value in quarters.items() if name != inked)
    assert quarters[inked] < 0.6 * blank, quarters


def test_ink_takes_the_pen_colour():
    texels = np.full((20, 20, 3), viewer.CHANNEL_MAX, dtype=np.uint8)
    viewer.ink(texels, (10.0, 2.0), (10.0, 17.0), CYAN.rgb)

    assert (texels[9:12, 1:19] == CYAN.rgb).all()
    assert (texels[:9] == viewer.CHANNEL_MAX).all()
    assert (texels[12:] == viewer.CHANNEL_MAX).all()


def wait_for_status(status: zmq.Socket, publish) -> None:
    """Publish until the subscription has reached the publisher."""
    deadline = time.monotonic() + RECEIVE_TIMEOUT_S
    while not status.poll(POLL_MS):
        assert time.monotonic() < deadline, "no pen status arrived"
        publish()


def test_the_newest_pen_announcement_wins():
    endpoint = "inproc://viewer-pen-status-test"
    with (
        closing(PenAnnouncer(endpoint)) as announcer,
        viewer.subscribe_pen_status(endpoint) as status,
    ):
        assert viewer.latest_pen_rgb(status) is None
        wait_for_status(status, lambda: announcer.announce(CYAN))
        announcer.announce(MAGENTA)

        assert viewer.latest_pen_rgb(status) == MAGENTA.rgb
        assert viewer.latest_pen_rgb(status) is None
    assert viewer.pen_rgba(MAGENTA.rgb) == pytest.approx(
        (236 / 255, 0.0, 140 / 255, 1.0)
    )


def test_a_malformed_pen_status_is_refused():
    endpoint = "inproc://viewer-malformed-status-test"
    publisher = zmq.Context.instance().socket(zmq.PUB)
    publisher.bind(endpoint)
    try:
        with viewer.subscribe_pen_status(endpoint) as status:
            wait_for_status(
                status, lambda: publisher.send_json({"name": "cyan", "rgb": [0, 174]})
            )
            with pytest.raises(ValueError, match="three integers"):
                viewer.latest_pen_rgb(status)
    finally:
        publisher.close(linger=0)


def dressed_spec(tmp_path: Path, *links: str) -> tuple[mujoco.MjSpec, Path]:
    write_box_stl(tmp_path / "assets" / "anchor.stl", (2.0, 4.0, 8.0))
    write_box_stl(tmp_path / "covers" / "cover.stl", (1.0, 3.0, 6.0))
    bundle = tmp_path / "bundle.xml"
    bundle.write_text(DRESSED_BUNDLE)
    urdf = tmp_path / "robot" / "model.urdf"
    urdf.parent.mkdir()
    urdf.write_text(
        '<robot name="dressed">'
        '<material name="orange"><color rgba="0.9 0.4 0.1 1"/></material>'
        f"{''.join(links)}</robot>"
    )
    return mujoco.MjSpec.from_file(str(bundle)), urdf


def mesh_geom(model: mujoco.MjModel, mesh: str) -> int:
    (geom,) = np.flatnonzero(model.geom_dataid == model.mesh(mesh).id)
    return int(geom)


def test_dressing_colours_meshes_and_places_missing_visuals(tmp_path):
    spec, urdf = dressed_spec(tmp_path, FLANGE_LINK)
    viewer.dress_robot(spec, urdf)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    anchor = mesh_geom(model, "anchor")
    assert model.geom_rgba[anchor] == pytest.approx([0.9, 0.4, 0.1, 1.0])

    # The bundle puts the anchor mesh 0.1 m along the flange body, where the URDF
    # has it 0.05 m from the link frame, so the link frame sits 0.05 m along the
    # body at (0.3, 0, 0.5) and the cover at 0.2 m along its Y, turned 90 degrees.
    cover = mesh_geom(model, "urdf_covers_cover")
    assert model.geom_rgba[cover] == pytest.approx([0.1, 0.1, 0.1, 1.0])
    assert data.geom_xpos[cover] == pytest.approx([0.35, 0.2, 0.5], abs=1e-6)
    rotation = data.geom_xmat[cover].reshape(3, 3)
    world_half_sizes = np.abs(rotation) @ model.geom_aabb[cover][3:]
    assert world_half_sizes == pytest.approx([0.003, 0.001, 0.006], abs=1e-6)


def test_dressing_rejects_a_visual_it_cannot_anchor(tmp_path):
    spec, urdf = dressed_spec(tmp_path, FLANGE_LINK, ORPHAN_LINK)
    with pytest.raises(ValueError, match="URDF link orphan"):
        viewer.dress_robot(spec, urdf)
