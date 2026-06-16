from pathlib import Path
import mujoco

# 처음에 "mujoco_menagerie"로 했다가 오류나서 Path("mujoco_menagerie") 로 수정
MENAGERIE_DIR = Path("mujoco_menagerie")
PANDA_XML = MENAGERIE_DIR / "franka_emika_panda" / "panda.xml"

model = mujoco.MjModel.from_xml_path(str(PANDA_XML))
# 현재 시뮬레이션 상태를 담는 data 객체 생성
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("Joint positions")
print("-" * 60)
print(f"{'joint_id':<10} {'name':<25} {'qpos_index':<12} {'position':<12}")
print("-" * 60)

# 모든 조인트에 대해 반복
for joint_id in range(model.njnt):
    # 조인트 ID를 조인트 이름으로 변환
    joint_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id
    )
    # 해당 조인트의 위치 값이 data.qpos의 몇 번째에 저장되는지 확인
    qpos_index = model.jnt_qposadr[joint_id]
    
    # qpos에서 실제 위치 값 읽기
    position = data.qpos[qpos_index]

    print(f"{joint_id:<10} {joint_name:<25} {qpos_index:<12} {position:>12.6f}")