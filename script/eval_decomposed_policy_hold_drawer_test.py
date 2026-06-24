import sys
import os
import subprocess
import gc
import signal

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import json

from generate_episode_instructions import *


current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def base_task_name_for_limit(task_name):
    suffixes = ("_decomposition",)
    for suffix in suffixes:
        if task_name.endswith(suffix):
            return task_name[: -len(suffix)]
    return task_name


def get_eval_step_limit(task_name):
    step_limit_path = os.path.join(parent_directory, "../task_config/_eval_step_limit.yml")
    with open(step_limit_path, "r", encoding="utf-8") as f:
        step_limits = yaml.safe_load(f)

    candidates = [task_name, base_task_name_for_limit(task_name)]
    for candidate in candidates:
        if candidate in step_limits:
            return int(step_limits[candidate]), candidate
    return 1000, None


def apply_eval_step_limit(task_env, task_name):
    step_limit, matched_name = get_eval_step_limit(task_name)
    task_env.step_lim = step_limit
    return step_limit, matched_name


def natural_subtask_key(path):
    name = Path(path).name
    marker = "_subtask"
    if marker not in name:
        return (999999, name)
    tail = name.rsplit(marker, 1)[-1]
    digits = ""
    for char in tail:
        if char.isdigit():
            digits += char
        else:
            break
    return (int(digits) if digits else 999999, name)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("1", "true", "yes", "y", "on"):
            return True
        if value in ("0", "false", "no", "n", "off", "none"):
            return False
    return bool(value)


def find_subtask_ckpt_dirs(args, usr_args):
    ckpt_root = Path(usr_args.get("ckpt_root", "policy/ACT/act_ckpt"))
    task_name = args["task_name"]
    ckpt_setting = args["ckpt_setting"]
    expert_data_num = usr_args.get("expert_data_num")
    explicit_count = usr_args.get("subtask_count")

    candidates = []
    if explicit_count is not None:
        subtask_count = int(explicit_count)
        for sub_id in range(1, subtask_count + 1):
            policy_task_name = f"{task_name}_subtask{sub_id}"
            task_ckpt_root = ckpt_root / f"act-{policy_task_name}"
            subtask_expert_data_num = get_subtask_expert_data_num(usr_args, sub_id, expert_data_num)
            ckpt_dir = resolve_subtask_ckpt_dir(task_ckpt_root, ckpt_setting, subtask_expert_data_num)
            candidates.append(
                {
                    "subtask_id": sub_id,
                    "policy_task_name": policy_task_name,
                    "expert_data_num": subtask_expert_data_num,
                    "ckpt_dir": str(ckpt_dir),
                }
            )
        return candidates

    pattern = f"act-{task_name}_subtask*"
    for task_ckpt_root in sorted(ckpt_root.glob(pattern), key=natural_subtask_key):
        if not task_ckpt_root.is_dir():
            continue
        subtask_id = natural_subtask_key(task_ckpt_root)[0]
        subtask_expert_data_num = get_subtask_expert_data_num(usr_args, subtask_id, expert_data_num)
        ckpt_dir = resolve_subtask_ckpt_dir(task_ckpt_root, ckpt_setting, subtask_expert_data_num)
        candidates.append(
            {
                "subtask_id": subtask_id,
                "policy_task_name": task_ckpt_root.name[len("act-"):],
                "expert_data_num": subtask_expert_data_num,
                "ckpt_dir": str(ckpt_dir),
            }
        )

    if not candidates:
        raise FileNotFoundError(
            f"No subtask checkpoints found under {ckpt_root} with pattern {pattern}"
        )
    return candidates


def get_subtask_expert_data_num(usr_args, subtask_id, default_expert_data_num=None):
    value = usr_args.get(f"subtask{subtask_id}_expert_data_num")
    if value is None:
        value = default_expert_data_num
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return value


def resolve_subtask_ckpt_dir(task_ckpt_root, ckpt_setting, expert_data_num=None):
    task_ckpt_root = Path(task_ckpt_root)
    if expert_data_num is not None:
        ckpt_dir = task_ckpt_root / f"{ckpt_setting}-{expert_data_num}"
        require_act_ckpt_dir(ckpt_dir)
        return ckpt_dir

    preferred = sorted(task_ckpt_root.glob(f"{ckpt_setting}-*"))
    for ckpt_dir in preferred:
        if is_act_ckpt_dir(ckpt_dir):
            return ckpt_dir

    for ckpt_dir in sorted(task_ckpt_root.iterdir() if task_ckpt_root.exists() else []):
        if is_act_ckpt_dir(ckpt_dir):
            return ckpt_dir

    require_act_ckpt_dir(task_ckpt_root)
    return task_ckpt_root


def is_act_ckpt_dir(path):
    path = Path(path)
    return path.is_dir() and (path / "dataset_stats.pkl").is_file() and (path / "policy_last.ckpt").is_file()


def require_act_ckpt_dir(path):
    if not is_act_ckpt_dir(path):
        raise FileNotFoundError(
            f"Invalid ACT checkpoint dir: {path}. Expected dataset_stats.pkl and policy_last.ckpt."
        )


def update_subtask_state(task_env, subtask_id, success):
    index = int(subtask_id) - 1
    success = bool(success)
    if hasattr(task_env, "subtask_success") and index < len(task_env.subtask_success):
        task_env.subtask_success[index] = success
    if hasattr(task_env, "subtask_done") and index < len(task_env.subtask_done):
        task_env.subtask_done[index] = True


def reset_decomposition_eval_state(task_env):
    if hasattr(task_env, "subtask_success"):
        task_env.subtask_success = [False for _ in task_env.subtask_success]
    if hasattr(task_env, "subtask_done"):
        task_env.subtask_done = [False for _ in task_env.subtask_done]
    if hasattr(task_env, "current_subtask"):
        task_env.current_subtask = 0
    task_env.eval_success = False


def check_subtask_success(task_env, subtask_id, is_final):
    func_name = f"_check_subtask{subtask_id}_success"
    if hasattr(task_env, func_name):
        success = bool(getattr(task_env, func_name)())
        update_subtask_state(task_env, subtask_id, success)
        if success:
            return True

    if is_final:
        try:
            return bool(task_env.check_success())
        except Exception:
            return False
    return False


def build_model_args(usr_args, subtask_info):
    model_args = dict(usr_args)
    model_args["task_name"] = subtask_info["policy_task_name"]
    model_args["ckpt_dir"] = subtask_info["ckpt_dir"]
    return model_args


def generate_instruction_for_episode(task_name, episode_info, test_num, instruction_type):
    episode_info_list = [episode_info.get("info", {})]
    candidate_names = [task_name]
    if task_name.endswith("_decomposition"):
        candidate_names.append(task_name[: -len("_decomposition")])

    for candidate_name in candidate_names:
        try:
            results = generate_episode_descriptions(candidate_name, episode_info_list, test_num)
            return np.random.choice(results[0][instruction_type])
        except Exception:
            continue
    return None


def unload_model(model):
    try:
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        gc.collect()


def call_with_timeout(func, timeout_sec, *args, **kwargs):
    timeout_sec = float(timeout_sec or 0)
    sigalrm = getattr(signal, "SIGALRM", None)
    if timeout_sec <= 0 or sigalrm is None:
        return func(*args, **kwargs)

    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"action eval exceeded {timeout_sec} seconds")

    old_handler = signal.signal(sigalrm, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(sigalrm, old_handler)


def load_resume_records(save_dir):
    save_dir = Path(save_dir)
    jsonl_path = save_dir / "episode_records.jsonl"
    json_path = save_dir / "episode_records.json"
    records = []

    if jsonl_path.is_file():
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: skip invalid jsonl record in {jsonl_path}")
        return records

    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("episodes"), list):
            return data["episodes"]
    return records


def append_episode_record(jsonl_path, record):
    jsonl_path = Path(jsonl_path)
    with jsonl_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def build_run_metadata(task_name,
                       task_config,
                       ckpt_setting,
                       policy_name,
                       instruction_type,
                       subtask_infos,
                       official_step_limit,
                       official_step_limit_name,
                       default_subtask_step_limit,
                       test_num):
    if default_subtask_step_limit is None:
        budget_mode = "shared_official_total_budget"
        per_subtask_step_limit = None
    else:
        budget_mode = "fixed_per_subtask_budget"
        per_subtask_step_limit = int(default_subtask_step_limit)

    return {
        "task_name": task_name,
        "task_config": task_config,
        "ckpt_setting": ckpt_setting,
        "policy_name": policy_name,
        "instruction_type": instruction_type,
        "test_num": int(test_num),
        "official_step_limit": int(official_step_limit),
        "official_step_limit_matched_task": official_step_limit_name,
        "budget_mode": budget_mode,
        "per_subtask_step_limit": per_subtask_step_limit,
        "success_budget_condition": (
            "sum of ACT action steps across subtasks must be <= official_step_limit"
            if default_subtask_step_limit is None
            else "each subtask uses the explicit per_subtask_step_limit"
        ),
        "subtask_checkpoints": subtask_infos,
    }


def write_episode_records_json(detail_path, metadata, episode_records):
    detail_path = Path(detail_path)
    payload = {
        "metadata": metadata,
        "summary": {
            "completed_episodes": len(episode_records),
            "successes": sum(1 for record in episode_records if record.get("success")),
            "success_rate": (
                sum(1 for record in episode_records if record.get("success")) / len(episode_records)
                if episode_records else 0.0
            ),
        },
        "episodes": episode_records,
    }
    with detail_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_eval_video_frame(task_env, refresh=True):
    if getattr(task_env, "eval_video_path", None) is None:
        return False
    if not getattr(task_env, "eval_video_ffmpeg", None):
        return False
    if refresh:
        obs = task_env.get_obs()
        frame = obs["observation"]["head_camera"]["rgb"]
    else:
        task_env.cameras.update_picture()
        frame = task_env.cameras.get_rgb()["head_camera"]["rgb"]
    task_env.eval_video_ffmpeg.stdin.write(frame.tobytes())
    return True


def _arm_tag_name(arm_tag):
    return str(arm_tag).lower()


def _get_arm_qpos_and_gripper(task_env, arm_tag):
    arm_name = _arm_tag_name(arm_tag)
    if arm_name == "left":
        joint_state = task_env.robot.get_left_arm_jointState()
        return np.asarray(joint_state[:-1], dtype=np.float64), float(task_env.robot.get_left_gripper_val())
    if arm_name == "right":
        joint_state = task_env.robot.get_right_arm_jointState()
        return np.asarray(joint_state[:-1], dtype=np.float64), float(task_env.robot.get_right_gripper_val())
    raise ValueError(f"unknown arm tag for hold drawer: {arm_tag}")


def _overwrite_action_arm(action, arm_tag, locked_qpos, locked_gripper):
    action_np = np.array(action, copy=True)
    if action_np.shape[-1] < 14:
        raise ValueError(f"hold drawer expects 14-dim action, got shape={action_np.shape}")

    arm_name = _arm_tag_name(arm_tag)
    if arm_name == "left":
        action_np[..., 0:6] = locked_qpos
        action_np[..., 6] = locked_gripper
    elif arm_name == "right":
        action_np[..., 7:13] = locked_qpos
        action_np[..., 13] = locked_gripper
    else:
        raise ValueError(f"unknown arm tag for hold drawer: {arm_tag}")
    return action_np


def prepare_hold_drawer_state(task_env, hold_drawer_gripper_pos=None, require_contact=True):
    state = {
        "enabled": True,
        "arm": None,
        "verified": False,
        "locked_qpos": None,
        "locked_gripper": None,
        "action_mask_count": 0,
        "error": None,
    }

    if not hasattr(task_env, "arm_tag"):
        state["error"] = "task_env.arm_tag is missing before drawer-hold snapshot"
        return state

    drawer_arm_tag = task_env.arm_tag.opposite
    state["arm"] = _arm_tag_name(drawer_arm_tag)

    try:
        qpos, current_gripper = _get_arm_qpos_and_gripper(task_env, drawer_arm_tag)
        state["locked_qpos"] = qpos.tolist()
        state["locked_gripper"] = float(
            current_gripper if hold_drawer_gripper_pos is None else hold_drawer_gripper_pos
        )

        contact_verified = None
        if hasattr(task_env, "_drawer_arm_holding_cabinet"):
            contact_verified = bool(task_env._drawer_arm_holding_cabinet(drawer_arm_tag))
        if contact_verified is None:
            contact_verified = bool(
                task_env.is_left_gripper_close()
                if state["arm"] == "left"
                else task_env.is_right_gripper_close()
            )
        state["verified"] = bool(contact_verified)
        if require_contact and not state["verified"]:
            state["error"] = "drawer arm did not hold cabinet handle at subtask1 success"
        elif not require_contact and not state["verified"]:
            state["error"] = "drawer arm hold contact was not verified; continuing because require_contact=false"
    except Exception as e:
        state["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()

    return state


def build_hold_drawer_take_action(original_take_action, hold_state, debug_actions=False, task_env=None, subtask_id=None):
    drawer_arm_tag = hold_state["arm"]
    locked_qpos = np.asarray(hold_state["locked_qpos"], dtype=np.float64)
    locked_gripper = float(hold_state["locked_gripper"])

    def hold_drawer_take_action(action, *take_args, **take_kwargs):
        masked_action = _overwrite_action_arm(action, drawer_arm_tag, locked_qpos, locked_gripper)
        hold_state["action_mask_count"] = int(hold_state.get("action_mask_count", 0)) + 1
        if debug_actions:
            print(
                f"[hold_drawer_action] subtask{subtask_id} "
                f"cnt={getattr(task_env, 'take_action_cnt', '?')}/{getattr(task_env, 'step_lim', '?')} "
                f"arm={drawer_arm_tag} masks={hold_state['action_mask_count']}",
                flush=True,
            )
        return original_take_action(masked_action, *take_args, **take_kwargs)

    return hold_drawer_take_action


def run_reset_after_subtask(task_env, reset_video_freq=10, subtask_id=None):
    if not hasattr(task_env, "_reset_after_subtask"):
        return {"called": False, "success": True, "video_frames": 0, "error": None}

    result = {"called": True, "success": False, "video_frames": 0, "error": None}
    reset_video_freq = max(1, int(reset_video_freq))
    original_take_dense_action = task_env.take_dense_action
    original_update_render = task_env._update_render
    original_take_picture = task_env._take_picture

    def update_render_with_video():
        out = original_update_render()
        if write_eval_video_frame(task_env, refresh=False):
            result["video_frames"] += 1
        return out

    def take_dense_action_with_video(control_seq, save_freq=-1):
        return original_take_dense_action(control_seq, save_freq=reset_video_freq)

    try:
        task_env._update_render = update_render_with_video
        task_env._take_picture = lambda: None
        task_env.take_dense_action = take_dense_action_with_video
        try:
            reset_return = task_env._reset_after_subtask(subtask_id=subtask_id)
        except TypeError:
            reset_return = task_env._reset_after_subtask()
        result["success"] = reset_return is not False
        if not result["success"]:
            result["error"] = "reset_after_subtask returned False"
    except Exception as e:
        result["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
    finally:
        task_env.take_dense_action = original_take_dense_action
        task_env._update_render = original_update_render
        task_env._take_picture = original_take_picture
    return result


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    hold_drawer_test_enabled = parse_bool(usr_args.get("hold_drawer_arm_subtask2", False))
    if hold_drawer_test_enabled:
        args["eval_video_log"] = True
        if parse_bool(usr_args.get("disable_eval_video_log", False)):
            print(
                "[HOLD_DRAWER] ignore --disable_eval_video_log because hold-drawer test needs video",
                flush=True,
            )
    elif parse_bool(usr_args.get("disable_eval_video_log", False)):
        args["eval_video_log"] = False

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    resume_result_dir = usr_args.get("resume_result_dir")
    if resume_result_dir:
        save_dir = Path(resume_result_dir)
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}_decomposed/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)
    episode_jsonl_path = save_dir / "episode_records.jsonl"
    resume_records = load_resume_records(save_dir) if resume_result_dir else []
    if resume_records:
        print(f"\033[93mResume:\033[0m loaded {len(resume_records)} finished episode records from {save_dir}")

    if args["eval_video_log"]:
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        args["eval_video_save_dir"] = save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    subtask_infos = find_subtask_ckpt_dirs(args, usr_args)
    print("\033[94mSubtask checkpoints:\033[0m")
    for subtask_info in subtask_infos:
        print(
            f" - subtask{subtask_info['subtask_id']} "
            f"(expert_data_num={subtask_info.get('expert_data_num')}): "
            f"{subtask_info['ckpt_dir']}"
        )

    seed = usr_args["seed"]
    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))
    topk = 1
    official_step_limit, official_step_limit_name = get_eval_step_limit(args["task_name"])
    default_subtask_step_limit = usr_args.get("subtask_step_limit")
    if isinstance(default_subtask_step_limit, str) and default_subtask_step_limit.lower() == "none":
        default_subtask_step_limit = None
    if default_subtask_step_limit is not None:
        default_subtask_step_limit = int(default_subtask_step_limit)
    run_metadata = build_run_metadata(
        task_name=args["task_name"],
        task_config=task_config,
        ckpt_setting=ckpt_setting,
        policy_name=policy_name,
        instruction_type=instruction_type,
        subtask_infos=subtask_infos,
        official_step_limit=official_step_limit,
        official_step_limit_name=official_step_limit_name,
        default_subtask_step_limit=default_subtask_step_limit,
        test_num=test_num,
    )
    detail_path = os.path.join(save_dir, "episode_records.json")
    metadata_path = os.path.join(save_dir, "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2, ensure_ascii=False)

    st_seed, suc_num, episode_records = eval_decomposed_policy(
        task_name,
        TASK_ENV,
        args,
        usr_args,
        get_model,
        subtask_infos,
        st_seed,
        save_dir=save_dir,
        episode_jsonl_path=episode_jsonl_path,
        detail_path=detail_path,
        run_metadata=run_metadata,
        resume_records=resume_records,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
    )
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, "_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Official Step Limit: {run_metadata['official_step_limit']}\n")
        file.write(f"Official Step Limit Matched Task: {run_metadata['official_step_limit_matched_task']}\n")
        file.write(f"Budget Mode: {run_metadata['budget_mode']}\n\n")
        file.write(f"Subtask Checkpoints:\n")
        for subtask_info in subtask_infos:
            file.write(
                f"  subtask{subtask_info['subtask_id']} "
                f"(expert_data_num={subtask_info.get('expert_data_num')}): "
                f"{subtask_info['ckpt_dir']}\n"
            )
        file.write("\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    write_episode_records_json(detail_path, run_metadata, episode_records)
    if os.path.exists(episode_jsonl_path):
        os.remove(episode_jsonl_path)
        print(f"Removed resume jsonl after successful completion: {episode_jsonl_path}")

    print(f"Data has been saved to {file_path}")
    print(f"Episode details have been saved to {detail_path}")


def eval_decomposed_policy(task_name,
                           TASK_ENV,
                           args,
                           usr_args,
                           get_model,
                           subtask_infos,
                           st_seed,
                           save_dir=None,
                           episode_jsonl_path=None,
                           detail_path=None,
                           run_metadata=None,
                           resume_records=None,
                           test_num=100,
                           video_size=None,
                           instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = parse_bool(usr_args.get("expert_check", True))
    resume_records = resume_records or []
    completed_num = len(resume_records)
    completed_success = sum(1 for record in resume_records if record.get("success"))
    TASK_ENV.suc = completed_success
    TASK_ENV.test_num = completed_num

    now_id = completed_num
    succ_seed = completed_num
    suc_test_seed_list = []
    episode_records = list(resume_records)

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    if resume_records:
        last_record = resume_records[-1]
        if last_record.get("next_seed") is not None:
            now_seed = int(last_record["next_seed"])
        else:
            now_seed = int(last_record.get("seed", st_seed)) + 1
        print(
            f"\033[93mResume:\033[0m already finished {completed_num}/{test_num}, "
            f"success {completed_success}/{completed_num}, continue from seed {now_seed}",
            flush=True,
        )
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True
    post_success_steps = int(usr_args.get("post_subtask_success_steps", 0))
    progress_log_freq = int(usr_args.get("progress_log_freq", 0) or 0)
    debug_actions = parse_bool(usr_args.get("debug_actions", False))
    action_timeout_sec = float(usr_args.get("action_timeout_sec", 0) or 0)
    no_progress_limit = int(usr_args.get("no_progress_limit", 3) or 0)
    reset_between_subtasks = parse_bool(usr_args.get("reset_between_subtasks", True))
    reset_video_freq = int(usr_args.get("reset_video_freq", 10) or 10)
    hold_drawer_enabled = parse_bool(usr_args.get("hold_drawer_arm_subtask2", False))
    hold_drawer_require_contact = parse_bool(usr_args.get("hold_drawer_require_contact", True))
    hold_drawer_gripper_pos_arg = usr_args.get("hold_drawer_gripper_pos")
    if isinstance(hold_drawer_gripper_pos_arg, str) and hold_drawer_gripper_pos_arg.strip().lower() in ("", "none", "null", "current"):
        hold_drawer_gripper_pos = None
    elif hold_drawer_gripper_pos_arg is None:
        hold_drawer_gripper_pos = None
    else:
        hold_drawer_gripper_pos = float(hold_drawer_gripper_pos_arg)
    default_subtask_step_limit = usr_args.get("subtask_step_limit")
    if isinstance(default_subtask_step_limit, str) and default_subtask_step_limit.lower() == "none":
        default_subtask_step_limit = None
    if default_subtask_step_limit is not None:
        default_subtask_step_limit = int(default_subtask_step_limit)

    official_step_limit, official_step_limit_name = get_eval_step_limit(args["task_name"])
    print(
        f"\033[94mOfficial eval step limit:\033[0m {official_step_limit} "
        f"(matched task: {official_step_limit_name or 'fallback_default'})"
    )
    if default_subtask_step_limit is None:
        print("\033[94mSubtask step limit:\033[0m shared official total budget")
    else:
        print(f"\033[94mSubtask step limit:\033[0m {default_subtask_step_limit} per subtask")

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        episode_info = {"info": {}}

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                apply_eval_step_limit(TASK_ENV, args["task_name"])
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        apply_eval_step_limit(TASK_ENV, args["task_name"])
        reset_decomposition_eval_state(TASK_ENV)
        instruction = generate_instruction_for_episode(
            args["task_name"],
            episode_info,
            test_num,
            instruction_type,
        )
        TASK_ENV.set_instruction(instruction=instruction)

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        phase_records = []
        total_subtasks = len(subtask_infos)
        hold_drawer_state_for_subtask2 = None

        for subtask_index, subtask_info in enumerate(subtask_infos):
            subtask_id = int(subtask_info["subtask_id"])
            is_final = subtask_index == total_subtasks - 1
            phase_record = {
                "subtask_id": subtask_id,
                "policy_task_name": subtask_info["policy_task_name"],
                "ckpt_dir": subtask_info["ckpt_dir"],
                "start_take_action_cnt": int(TASK_ENV.take_action_cnt),
                "official_step_limit": int(TASK_ENV.step_lim),
                "remaining_total_step_budget_at_start": max(0, int(TASK_ENV.step_lim) - int(TASK_ENV.take_action_cnt)),
                "success": False,
                "failure_reason": None,
                "runtime_error": None,
                "drawer_hold_enabled": False,
                "drawer_hold_arm": None,
                "drawer_hold_verified": None,
                "drawer_hold_locked_qpos": None,
                "drawer_hold_locked_gripper": None,
                "drawer_hold_action_mask_count": 0,
                "drawer_hold_error": None,
            }

            print(
                f"\033[93mRunning subtask{subtask_id}\033[0m | "
                f"\033[94m{subtask_info['policy_task_name']}\033[0m"
            )

            if hasattr(TASK_ENV, "_setup_subtask_config"):
                TASK_ENV._setup_subtask_config(subtask_id)
            TASK_ENV.eval_success = False

            model = get_model(build_model_args(usr_args, subtask_info))
            reset_func(model)
            original_take_action = TASK_ENV.take_action
            take_action_wrapped = False
            hold_drawer_state = None

            if (
                hold_drawer_enabled
                and args["task_name"] == "put_object_cabinet_decomposition"
                and subtask_id == 2
            ):
                hold_drawer_state = hold_drawer_state_for_subtask2
                if hold_drawer_state is None:
                    hold_drawer_state = prepare_hold_drawer_state(
                        TASK_ENV,
                        hold_drawer_gripper_pos=hold_drawer_gripper_pos,
                        require_contact=hold_drawer_require_contact,
                    )
                    hold_drawer_state["error"] = hold_drawer_state.get("error") or "drawer-hold snapshot was not captured at subtask1 success"
                phase_record["drawer_hold_enabled"] = True
                phase_record["drawer_hold_arm"] = hold_drawer_state.get("arm")
                phase_record["drawer_hold_verified"] = bool(hold_drawer_state.get("verified"))
                phase_record["drawer_hold_locked_qpos"] = hold_drawer_state.get("locked_qpos")
                phase_record["drawer_hold_locked_gripper"] = hold_drawer_state.get("locked_gripper")
                phase_record["drawer_hold_error"] = hold_drawer_state.get("error")

                if not hold_drawer_state.get("verified") and hold_drawer_require_contact:
                    print(
                        "[HOLD_DRAWER_FAIL] drawer arm did not hold cabinet handle at subtask1 success",
                        flush=True,
                    )
                    phase_record["success"] = False
                    phase_record["failure_reason"] = "drawer_hold_not_verified_at_subtask1_success"
                    phase_record["runtime_error"] = None
                    phase_record["end_take_action_cnt"] = int(TASK_ENV.take_action_cnt)
                    phase_record["used_steps"] = phase_record["end_take_action_cnt"] - phase_record["start_take_action_cnt"]
                    phase_record["remaining_total_step_budget_at_end"] = max(0, int(TASK_ENV.step_lim) - int(TASK_ENV.take_action_cnt))
                    phase_record["reset_after_subtask"] = None
                    phase_records.append(phase_record)
                    unload_model(model)
                    break

                print(
                    f"[HOLD_DRAWER] subtask2 masks {hold_drawer_state.get('arm')} arm "
                    f"using subtask1-success snapshot verified={hold_drawer_state.get('verified')} "
                    f"locked_gripper={hold_drawer_state.get('locked_gripper')}",
                    flush=True,
                )
                TASK_ENV.take_action = build_hold_drawer_take_action(
                    original_take_action,
                    hold_drawer_state,
                    debug_actions=debug_actions,
                    task_env=TASK_ENV,
                    subtask_id=subtask_id,
                )
                take_action_wrapped = True
            elif debug_actions:
                def logged_take_action(action, *take_args, **take_kwargs):
                    action_np = np.asarray(action)
                    finite = bool(np.all(np.isfinite(action_np)))
                    action_min = float(np.nanmin(action_np)) if action_np.size else 0.0
                    action_max = float(np.nanmax(action_np)) if action_np.size else 0.0
                    print(
                        f"[debug_action] subtask{subtask_id} "
                        f"cnt={TASK_ENV.take_action_cnt}/{TASK_ENV.step_lim} "
                        f"shape={tuple(action_np.shape)} finite={finite} "
                        f"min={action_min:.5f} max={action_max:.5f}",
                        flush=True,
                    )
                    return original_take_action(action, *take_args, **take_kwargs)

                TASK_ENV.take_action = logged_take_action
                take_action_wrapped = True
            subtask_start_cnt = int(TASK_ENV.take_action_cnt)
            subtask_step_limit = default_subtask_step_limit
            if subtask_step_limit is None:
                subtask_step_limit = max(0, int(TASK_ENV.step_lim) - subtask_start_cnt)
            subtask_deadline = subtask_start_cnt + subtask_step_limit
            success_seen_at = None
            no_progress_count = 0

            try:
                while TASK_ENV.take_action_cnt < subtask_deadline and TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                    before_action_cnt = int(TASK_ENV.take_action_cnt)
                    observation = TASK_ENV.get_obs()
                    if progress_log_freq > 0 and TASK_ENV.take_action_cnt % progress_log_freq == 0:
                        print(
                            f"[progress] before eval subtask{subtask_id} "
                            f"cnt={TASK_ENV.take_action_cnt}/{TASK_ENV.step_lim}",
                            flush=True,
                        )
                    call_with_timeout(
                        eval_func,
                        action_timeout_sec,
                        TASK_ENV,
                        model,
                        observation,
                    )
                    if progress_log_freq > 0 and TASK_ENV.take_action_cnt % progress_log_freq == 0:
                        print(
                            f"[progress] after eval subtask{subtask_id} "
                            f"cnt={TASK_ENV.take_action_cnt}/{TASK_ENV.step_lim}",
                            flush=True,
                        )
                    subtask_success = check_subtask_success(TASK_ENV, subtask_id, is_final)
                    if not subtask_success:
                        TASK_ENV.eval_success = False
                    if int(TASK_ENV.take_action_cnt) == before_action_cnt and not subtask_success:
                        no_progress_count += 1
                        if no_progress_limit > 0 and no_progress_count >= no_progress_limit:
                            raise RuntimeError(
                                f"take_action made no progress for {no_progress_count} consecutive eval calls "
                                f"at subtask{subtask_id}, cnt={TASK_ENV.take_action_cnt}, "
                                f"eval_success={getattr(TASK_ENV, 'eval_success', None)}"
                            )
                    else:
                        no_progress_count = 0
                    if subtask_success and success_seen_at is None:
                        success_seen_at = int(TASK_ENV.take_action_cnt)
                    if subtask_success:
                        if is_final or post_success_steps <= 0:
                            phase_record["success"] = True
                            break
                        if TASK_ENV.take_action_cnt - success_seen_at >= post_success_steps:
                            phase_record["success"] = True
                            break
                else:
                    phase_record["success"] = check_subtask_success(TASK_ENV, subtask_id, is_final)
            except Exception as e:
                phase_record["runtime_error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
                phase_record["failure_reason"] = "runtime_error"
            finally:
                if take_action_wrapped:
                    TASK_ENV.take_action = original_take_action

            if hold_drawer_state is not None:
                phase_record["drawer_hold_action_mask_count"] = int(hold_drawer_state.get("action_mask_count", 0))

            phase_record["end_take_action_cnt"] = int(TASK_ENV.take_action_cnt)
            phase_record["used_steps"] = phase_record["end_take_action_cnt"] - phase_record["start_take_action_cnt"]
            phase_record["remaining_total_step_budget_at_end"] = max(0, int(TASK_ENV.step_lim) - int(TASK_ENV.take_action_cnt))
            phase_record["reset_after_subtask"] = None

            if not phase_record["success"] and phase_record["failure_reason"] is None:
                if int(TASK_ENV.take_action_cnt) >= int(TASK_ENV.step_lim):
                    phase_record["failure_reason"] = "official_total_step_limit_reached"
                elif int(TASK_ENV.take_action_cnt) >= int(subtask_deadline):
                    phase_record["failure_reason"] = "subtask_step_limit_reached"
                else:
                    phase_record["failure_reason"] = "subtask_success_condition_not_met"

            if not phase_record["success"] and hasattr(TASK_ENV, "get_subtask_debug_info"):
                phase_record["debug_info"] = TASK_ENV.get_subtask_debug_info(subtask_id)
                print(
                    f"[subtask_debug] {json.dumps(phase_record['debug_info'], ensure_ascii=False)}",
                    flush=True,
                )

            if (
                phase_record["success"]
                and hold_drawer_enabled
                and args["task_name"] == "put_object_cabinet_decomposition"
                and subtask_id == 1
                and total_subtasks >= 2
            ):
                hold_drawer_state_for_subtask2 = prepare_hold_drawer_state(
                    TASK_ENV,
                    hold_drawer_gripper_pos=hold_drawer_gripper_pos,
                    require_contact=hold_drawer_require_contact,
                )
                phase_record["drawer_hold_enabled"] = True
                phase_record["drawer_hold_arm"] = hold_drawer_state_for_subtask2.get("arm")
                phase_record["drawer_hold_verified"] = bool(hold_drawer_state_for_subtask2.get("verified"))
                phase_record["drawer_hold_locked_qpos"] = hold_drawer_state_for_subtask2.get("locked_qpos")
                phase_record["drawer_hold_locked_gripper"] = hold_drawer_state_for_subtask2.get("locked_gripper")
                phase_record["drawer_hold_error"] = hold_drawer_state_for_subtask2.get("error")
                if not hold_drawer_state_for_subtask2.get("verified") and hold_drawer_require_contact:
                    phase_record["success"] = False
                    phase_record["failure_reason"] = "drawer_hold_not_verified_at_subtask1_success"
                    phase_record["runtime_error"] = hold_drawer_state_for_subtask2.get("error")
                    print(
                        "[HOLD_DRAWER_FAIL] drawer arm snapshot at subtask1 success was not verified",
                        flush=True,
                    )
                else:
                    print(
                        f"[HOLD_DRAWER] captured subtask1-success snapshot for "
                        f"{hold_drawer_state_for_subtask2.get('arm')} arm "
                        f"verified={hold_drawer_state_for_subtask2.get('verified')} "
                        f"locked_gripper={hold_drawer_state_for_subtask2.get('locked_gripper')}",
                        flush=True,
                    )

            skip_transition_reset_for_hold_drawer = (
                hold_drawer_enabled
                and args["task_name"] == "put_object_cabinet_decomposition"
                and subtask_id == 1
                and total_subtasks >= 2
            )
            if phase_record["success"] and (not is_final) and reset_between_subtasks and not skip_transition_reset_for_hold_drawer:
                print(
                    f"\033[93mResetting arms after subtask{subtask_id}\033[0m",
                    flush=True,
                )
                phase_record["reset_after_subtask"] = run_reset_after_subtask(
                    TASK_ENV,
                    reset_video_freq=reset_video_freq,
                    subtask_id=subtask_id,
                )
                if not phase_record["reset_after_subtask"]["success"]:
                    phase_record["success"] = False
                    phase_record["failure_reason"] = "reset_after_subtask_failed"
                    phase_record["runtime_error"] = (
                        phase_record["reset_after_subtask"]["error"]
                        or "reset_after_subtask failed"
                    )
            elif skip_transition_reset_for_hold_drawer:
                phase_record["reset_after_subtask"] = {
                    "called": False,
                    "success": bool(hold_drawer_state_for_subtask2 and hold_drawer_state_for_subtask2.get("verified")),
                    "video_frames": 0,
                    "error": "using_subtask1_success_snapshot_for_subtask2_action_mask",
                }

            phase_records.append(phase_record)
            unload_model(model)

            if not phase_record["success"]:
                break

        succ = bool(phase_records and all(record["success"] for record in phase_records))
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        episode_record = {
            "episode_id": now_id,
            "seed": now_seed,
            "next_seed": now_seed + 1,
            "success": succ,
            "official_step_limit": int(TASK_ENV.step_lim),
            "total_used_steps": int(TASK_ENV.take_action_cnt),
            "remaining_total_step_budget": max(0, int(TASK_ENV.step_lim) - int(TASK_ENV.take_action_cnt)),
            "instruction": str(instruction),
            "phases": phase_records,
        }
        episode_records.append(episode_record)
        if episode_jsonl_path is not None:
            append_episode_record(episode_jsonl_path, episode_record)
        if detail_path is not None and run_metadata is not None:
            write_episode_records_json(detail_path, run_metadata, episode_records)

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}_decomposed\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc, episode_records


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
