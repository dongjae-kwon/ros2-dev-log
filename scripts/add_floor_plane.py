from pathlib import Path
import mujoco

MENAGERIE_DIR = Path("mujoco_menagerie")
PANDA_XML = MENAGERIE_DIR / "franka_emika_panda" / "panda.xml"

base_xml = PANDA_XML.read_text(encoding="utf-8")

floor_xml = """
<geom name="floor"
      type="plane"
      pos="0 0 0"
      size="2 2 0.01"
      rgba="0.7 0.7 0.7 1"/>
"""

box_xml = """
<body name="object_box" pos="0.5 0 0.1">
    <freejoint name="object_freejoint"/>
    <geom name="object_box_geom"
          type="box"
          size="0.03 0.03 0.03"
          mass="0.1"
          rgba="0.8 0.2 0.2 1"/>
</body>
"""

modified_xml = base_xml.replace(
    "</worldbody>",
    floor_xml + "\n" + box_xml + "\n</worldbody>"
)

temp_xml_path = PANDA_XML.parent / "panda_modified.xml"
temp_xml_path.write_text(modified_xml, encoding="utf-8")

model = mujoco.MjModel.from_xml_path(temp_xml_path.as_posix())
data = mujoco.MjData(model)

while data.time < 1.0:
    mujoco.mj_step(model, data)

print("Simulation finished.")
print("Final time:", data.time)