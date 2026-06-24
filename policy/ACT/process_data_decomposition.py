import sys
sys.path.append("./policy/ACT/")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb
import json


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def pad_episode_to_min_chunks(qpos, actions, cam_high, cam_right_wrist, cam_left_wrist, left_arm_dim, right_arm_dim, min_chunk_num):
    if min_chunk_num <= len(qpos):
        return

    if len(qpos) == 0:
        raise ValueError("Cannot pad episode with zero frames. Ensure there is at least one valid frame available.")

    last_qpos = qpos[-1]
    last_action = actions[-1] if len(actions) > 0 else last_qpos
    last_cam_high = cam_high[-1]
    last_cam_right_wrist = cam_right_wrist[-1]
    last_cam_left_wrist = cam_left_wrist[-1]
    last_left_arm_dim = left_arm_dim[-1] if len(left_arm_dim) > 0 else 0
    last_right_arm_dim = right_arm_dim[-1] if len(right_arm_dim) > 0 else 0

    while len(qpos) < min_chunk_num:
        qpos.append(last_qpos)
        actions.append(last_action)
        cam_high.append(last_cam_high)
        cam_right_wrist.append(last_cam_right_wrist)
        cam_left_wrist.append(last_cam_left_wrist)
        left_arm_dim.append(last_left_arm_dim)
        right_arm_dim.append(last_right_arm_dim)


def data_transform(path, episode_num, save_path, subtask_id, min_chunk_num=0):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"
    

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 按顺序取前 N 个文件
    for i in range(episode_num):
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (load_hdf5(
            os.path.join(path, f"episode{i}.hdf5")))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            if j != left_gripper_all.shape[0] - 1:
                state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)  # joint

                state = state.astype(np.float32)
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        pad_episode_to_min_chunks(
            qpos,
            actions,
            cam_high,
            cam_right_wrist,
            cam_left_wrist,
            left_arm_dim,
            right_arm_dim,
            min_chunk_num,
        )

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            image.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
            image.create_dataset("cam_right_wrist", data=np.stack(cam_right_wrist), dtype=np.uint8)
            image.create_dataset("cam_left_wrist", data=np.stack(cam_left_wrist), dtype=np.uint8)

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process subtask episodes.")
    parser.add_argument("task_name", type=str, help="Task name")
    parser.add_argument("task_config", type=str, help="Task config")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to use")
    parser.add_argument("--subtask_id", type=int, required=True, choices=[1,2,3], help="Subtask ID (0 or 1)")
    parser.add_argument("--chunk_size", type=int, default=0, help="Minimum number of chunks per episode; pad with last frame if needed")

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num
    subtask_id = args.subtask_id
    min_chunk_num = args.chunk_size

    # 路径：data/task_name/task_config/data_subtask{subtask_id}
    data_path = os.path.join("../../data/", task_name, task_config, f"data_subtask{subtask_id}","data")
    
    # 输出路径
    save_path = f"processed_data/sim-{task_name}-subtask{subtask_id}/{task_config}-{expert_data_num}"
    
    # 处理数据
    begin = data_transform(data_path, expert_data_num, save_path, subtask_id, min_chunk_num)

    # 配置名：sim-taskname_subtask{i}-task_config-expert_data_num
    config_name = f"sim-{task_name}_subtask{subtask_id}-{task_config}-{expert_data_num}"
    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[config_name] = {
        "dataset_dir": f"./{save_path}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)

    print(f"✅ Saved config: {config_name}")