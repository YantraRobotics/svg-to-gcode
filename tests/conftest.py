import pytest

# The tool end of URDF_three_pen.urdf: three pens 120 degrees apart about joint 6.
THREE_PEN_URDF = b"""<robot name="owl">
  <link name="wrist_adapter_link"/>
  <link name="end_effector_link"/>
  <link name="three_pen_base"/>
  <link name="target_frame_1"/>
  <link name="target_frame_2"/>
  <link name="target_frame_3"/>
  <link name="finger"/>
  <joint name="joint6_ee_rotate" type="continuous">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="wrist_adapter_link"/>
    <child link="end_effector_link"/>
    <axis xyz="0.0 -1.0 0.0"/>
  </joint>
  <joint name="three_pen_mount" type="fixed">
    <origin xyz="0 -0.036 0" rpy="3.141592653589793 0 -1.57079632679"/>
    <parent link="end_effector_link"/>
    <child link="three_pen_base"/>
  </joint>
  <joint name="target_frame_1_joint" type="fixed">
    <origin xyz="0.277987 0 0.137979" rpy="0 -0.523598 0"/>
    <parent link="three_pen_base"/>
    <child link="target_frame_1"/>
  </joint>
  <joint name="target_frame_2_joint" type="fixed">
    <origin xyz="0.277987 0.119493 -0.06899" rpy="-2.034444 0.25268 0.463647"/>
    <parent link="three_pen_base"/>
    <child link="target_frame_2"/>
  </joint>
  <joint name="target_frame_3_joint" type="fixed">
    <origin xyz="0.277987 -0.119493 -0.06899" rpy="2.034444 0.25268 -0.463647"/>
    <parent link="three_pen_base"/>
    <child link="target_frame_3"/>
  </joint>
  <joint name="finger_joint" type="prismatic">
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <parent link="three_pen_base"/>
    <child link="finger"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
"""


@pytest.fixture
def three_pen_urdf() -> bytes:
    return THREE_PEN_URDF
