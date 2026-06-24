from envs import *
import random
import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
import yaml
import importlib
import json
import traceback
import time
from argparse import ArgumentParser
import sys
import os

# ================= 必须把路径设置放在最前面 =================
# 获取当前文件所在目录的上一级目录（即项目根目录），并加入系统路径
current_file_path = os.path.abspath(__file__)
root_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(root_directory)
sys.path.append("./")
# ==========================================================

# 设置好路径后，再去导包就不会报错了

# 此时这行代码就能完美读取项目根目录下的 envs 文件夹了，里面通常包含了 CONFIGS_PATH 等配置

sys.path.append("./")

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):
    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(
        CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(
        args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(
        args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(
            embodiment_type[0]) + "+" + str(embodiment_type[1])

    print("============= Config =============")
    print("\033[95mMessy Table:\033[0m " +
          str(args["domain_randomization"]["cluttered_table"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(
        args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    # 读取你 yml 里的 5 轮
    TARGET_NUM = args.get("episode_num", 5)
    SUBTASK_LIST = [1, 2]

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # ================= 解决 HDF5 报错的核心 =================
    # 强制创建好所有的子目录，防止 [Errno 2] 崩溃
    os.makedirs(args["save_path"], exist_ok=True)
    os.makedirs(os.path.join(args["save_path"],
                "data"), exist_ok=True)  # <--- 救命的就是这行！
    os.makedirs(os.path.join(args["save_path"], "video"), exist_ok=True)
    # ========================================================

    # =========== 阶段 1: 规划轨迹 (Planning) ===========
    if not args["use_seed"]:
        print("\033[93m[Phase 1: Planning 5 Episodes]\033[0m")
        args["need_plan"] = True
        seed_list = []
        fail_num = 0

        # 【核心改动】：使用 for 循环，雷打不动地推进 5 轮
        for epid in range(TARGET_NUM):
            try:
                TASK_ENV.setup_demo(now_ep_num=epid, seed=epid, **args)
                episode_success = True
                failed_subtask = None

                for sub_id in SUBTASK_LIST:
                    TASK_ENV._setup_subtask_config(sub_id)

                    try:
                        if sub_id == 1:
                            TASK_ENV._subtask1_place_can()
                        else:
                            TASK_ENV._subtask2_lift_basket()
                    except Exception as plan_error:
                        # 如果规划失败，只打印 Debug，不卡死！
                        print(
                            f"\033[93m[Debug] Planning Error at Round {epid}: Subtask {sub_id} condition failed. ({plan_error})\033[0m")

                    # 机械臂复原
                    TASK_ENV._reset_after_subtask()

                    if sub_id == 1:
                        success = TASK_ENV._check_subtask1_success()
                    else:
                        success = TASK_ENV._check_subtask2_success()

                    if not success:
                        episode_success = False
                        failed_subtask = sub_id if failed_subtask is None else failed_subtask

                    # 无条件保存轨迹
                    TASK_ENV.save_traj_data(f"{epid}_{sub_id}")

                if episode_success:
                    print(
                        f"simulate data episode {epid} success! (seed = {epid})")
                else:
                    print(
                        f"\033[91msimulate data episode {epid} fail! (seed = {epid}, Subtask {failed_subtask} failed)\033[0m")
                    fail_num += 1

                seed_list.append(epid)
                TASK_ENV.close_env()

            except Exception as e:
                print(f"\033[91mEpisode {epid} 彻底崩溃: {e}\033[0m")
                seed_list.append(epid)  # 崩溃了也记下 seed，凑够 5 个
                fail_num += 1
                TASK_ENV.close_env()

        with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
            for sed in seed_list:
                file.write("%s " % sed)
        print(
            f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {TARGET_NUM} tries \n")

   # =========== 阶段 2: 渲染数据与录制 (Collection) ===========
    if args.get("collect_data", True):
        print("\033[93m[Phase 2: Data Collection & Video Rendering]\033[0m")
        args["need_plan"] = False
        args["save_data"] = True
        args["render_freq"] = 0

        info_file_path = os.path.join(args["save_path"], "scene_info.json")

        # 【优化 1】：在内存中维护完整的 JSON 数据字典，不频繁读写硬盘
        info_db = {}

        for episode_idx in range(TARGET_NUM):
            print(f"\033[34mRendering Video for Episode {episode_idx}\033[0m")
            try:
                TASK_ENV.setup_demo(now_ep_num=episode_idx,
                                    seed=episode_idx, **args)
                episode_info = {}

                for sub_id in SUBTASK_LIST:
                    try:
                        TASK_ENV._setup_subtask_config(sub_id)
                        traj_data = TASK_ENV.load_tran_data(
                            f"{episode_idx}_{sub_id}")

                        # 安全获取轨迹数据，如果没有则默认为空列表
                        args["left_joint_path"] = traj_data.get(
                            "left_joint_path", [])
                        args["right_joint_path"] = traj_data.get(
                            "right_joint_path", [])

                        # 【优化 2】：拦截空轨迹！避免 list index out of range
                        if len(args["left_joint_path"]) == 0 and len(args["right_joint_path"]) == 0:
                            print(
                                f"\033[93m[Skip] Episode {episode_idx} Subtask {sub_id} 轨迹为空(规划失败)，跳过回放与录像。\033[0m")
                            episode_info[f"subtask{sub_id}_success"] = False
                            continue  # 直接跳过这个子任务的回放，进入下一个

                        TASK_ENV.set_path_lst(args)

                        # 回放动作
                        if sub_id == 1:
                            TASK_ENV._subtask1_place_can()
                            TASK_ENV._reset_after_subtask()
                        else:
                            TASK_ENV._subtask2_lift_basket()

                        # 记录状态并无条件保存 HDF5 和视频！
                        if sub_id == 1:
                            episode_info[f"subtask1_success"] = TASK_ENV._check_subtask1_success(
                            )
                        else:
                            episode_info[f"subtask2_success"] = TASK_ENV._check_subtask2_success(
                            )

                        TASK_ENV._save_subtask_data(f"_{sub_id}")

                    except Exception as subtask_error:
                        print(
                            f"\033[91mCollection Error at Ep {episode_idx} Sub {sub_id}: {subtask_error}\033[0m")
                        episode_info[f"subtask{sub_id}_success"] = False

                # 将当前这轮的结果存入内存字典
                info_db[f"episode_{episode_idx}"] = episode_info

                TASK_ENV.close_env(clear_cache=True)
            except Exception as e:
                print(f"渲染 Episode {episode_idx} 失败: {e}")
                TASK_ENV.close_env(clear_cache=True)

        # 【优化 1 收尾】：所有 5 轮结束后，一次性写入 JSON 文件，安全且高效
        try:
            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, indent=4, ensure_ascii=False)
            print(f"\033[92m[Success] scene_info.json 已安全保存！\033[0m")
        except Exception as e:
            print(f"\033[91m[Error] 保存 JSON 失败: {e}\033[0m")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()

    main(task_name=parser.task_name, task_config=parser.task_config)
