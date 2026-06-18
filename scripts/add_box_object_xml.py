from pathlib import Path
import mujoco

MENAGERIE_DIR = Path("mujoco_menagerie")
PANDA_XML = MENAGERIE_DIR / "franka_emika_panda" / "panda.xml"

base_xml = PANDA_XML.read_text(encoding="utf-8")

box_xml = """
<body name="object_box" pos="0.5 0 0.05">   
    <freejoint name="object_freejoint"/>
    <geom name="object_box_geom"
          type="box"
          size="0.03 0.03 0.03"
          mass="0.1"
          rgba="0.8 0.2 0.2 1"/>
</body>
"""

# 1. XML 문자열 변경
modified_xml = base_xml.replace(
    "</worldbody>",
    box_xml + "\n</worldbody>"
)

# 2. 원본 XML과 같은 폴더에 임시 파일로 저장
temp_xml_path = PANDA_XML.parent / "panda_modified.xml"
temp_xml_path.write_text(modified_xml, encoding="utf-8")

try:
    # 3. 파일 경로를 통해 로드 (상대 경로가 자동으로 유지되어 찾을 수 있음)
    model = mujoco.MjModel.from_xml_path(temp_xml_path.as_posix())
    data = mujoco.MjData(model)
    print("Model loaded with object box.")
    print("Number of bodies:", model.nbody)
    print("Number of joints:", model.njnt)
    print("Number of geoms:", model.ngeom)
finally:
    # 4. 로드가 끝난 후 임시 파일 삭제
    if temp_xml_path.exists():
        temp_xml_path.unlink()
