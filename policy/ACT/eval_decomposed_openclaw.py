#!/usr/bin/env python3
"""Run a decomposed RoboTwin ACT task with OpenClaw visual switching.

This follows RoboTwin's default ACT eval structure, but keeps one env alive:
subtask 1 is stopped by an OpenClaw switch decision, then subtask 2 is judged
with the original full-task check_success().
"""

from __future__ import annotations

import argparse
import base64
import gc
import importlib
import json
import os
import re
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import torch
import yaml

from decomposed_openclaw_bridge import DecomposedOpenClawBridge


ROBOTWIN_ROOT = Path("/data4/s2wxy/RoboTwin")
DEFAULT_ACT_CKPT_ROOT = ROBOTWIN_ROOT / "policy" / "ACT" / "act_ckpt"
DEFAULT_DECISION_BACKEND = "direct_llm"
DEFAULT_LLM_PROVIDER = "dmxapi"
DEFAULT_LLM_MODEL = "qwen3.6-plus"
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0
DEFAULT_LLM_OPENCLAW_CONFIG = "~/.openclaw/openclaw.json"
DEFAULT_LLM_API_KEY_ENV = "DMXAPI_API_KEY"
DEFAULT_LLM_IMAGE_MAX_EDGE = 768
DEFAULT_LLM_IMAGE_JPEG_QUALITY = 90
DEFAULT_LLM_MAX_TOKENS = 256
DEFAULT_LLM_RETRY_MAX_TOKENS = 512
DEFAULT_LLM_REASONING_EFFORT = "minimal"


@dataclass(frozen=True)
class SubtaskSpec:
    subtask_id: int
    name: str
    policy_task_name: str
    checkpoint_dir: str
    instruction: str
    success_condition: str
    visual_evidence: tuple[str, ...]


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    base_task_name: str
    env_module: str
    env_class: str
    task_config: str
    subtasks: tuple[SubtaskSpec, SubtaskSpec]


@dataclass(frozen=True)
class DirectLLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout_seconds: float
    image_max_edge: int
    image_jpeg_quality: int
    max_tokens: int
    retry_max_tokens: int
    reasoning_effort: str


TASKS: dict[str, TaskSpec] = {
    "place_can_basket_decomposition": TaskSpec(
        task_name="place_can_basket_decomposition",
        base_task_name="place_can_basket",
        env_module="envs.place_can_basket_decomposition",
        env_class="place_can_basket_decomposition",
        task_config="demo_random_light",
        subtasks=(
            SubtaskSpec(
                1,
                "put_can_in_basket",
                "place_can_basket_decomposition_subtask1",
                "/data4/s2wxy/RoboTwin/policy/ACT/act_ckpt/act-place_can_basket_decomposition_subtask1/demo_random_light-50",
                "Put the can into the basket.",
                "Subtask 1 is complete when the can is inside the basket.",
                (
                    "The can is visibly within the basket boundary from the camera views.",
                ),
            ),
            SubtaskSpec(
                2,
                "lift_basket",
                "place_can_basket_decomposition_subtask2",
                "/data4/s2wxy/RoboTwin/policy/ACT/act_ckpt/act-place_can_basket_decomposition_subtask2/demo_random_light-50",
                "Lift the basket after the can is inside.",
                "Final task success is judged by RoboTwin official check_success(): basket and can are lifted, basket upright, can stays near/in basket, can is not on table, and can contacts basket.",
                (
                    "Basket is lifted while staying upright.",
                    "Can remains with the basket and does not fall onto the table.",
                ),
            ),
        ),
    ),
    "place_burger_fries_decomposition": TaskSpec(
        task_name="place_burger_fries_decomposition",
        base_task_name="place_burger_fries",
        env_module="envs.place_burger_fries_decomposition",
        env_class="place_burger_fries_decomposition",
        task_config="demo_random_light",
        subtasks=(
            SubtaskSpec(
                1,
                "place_hamburger_on_tray",
                "place_burger_fries_decomposition_subtask1",
                "/data4/s2wxy/RoboTwin/policy/ACT/act_ckpt/act-place_burger_fries_decomposition_subtask1/demo_random_light-50",
                "Place the hamburger on the tray.",
                "Subtask 1 is complete when the hamburger is on the tray target area and the left gripper is open or no longer carrying it.",
                (
                    "Hamburger is visibly on the tray target area.",
                    "Left gripper is open or not holding the hamburger.",
                    "Hamburger is stable rather than mid-air.",
                ),
            ),
            SubtaskSpec(
                2,
                "place_fries_on_tray",
                "place_burger_fries_decomposition_subtask2",
                "/data4/s2wxy/RoboTwin/policy/ACT/act_ckpt/act-place_burger_fries_decomposition_subtask2/demo_random_light-50",
                "Place the fries on the tray.",
                "Final task success is judged by RoboTwin official check_success(): hamburger and fries are each within 8 cm of their tray target points and both grippers are open.",
                (
                    "Fries are on the tray target area.",
                    "Both hamburger and fries are on the tray.",
                    "Both grippers are open.",
                ),
            ),
        ),
    ),
}


class AnnotatedStdin:
    def __init__(self, raw_stdin, width: int, height: int, label_state: dict[str, str]):
        self.raw_stdin = raw_stdin
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self.label_state = label_state

    def write(self, data):
        if len(data) != self.frame_bytes:
            return self.raw_stdin.write(data)
        frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        label = self.label_state.get("label", "")[:64]
        overlay = frame.copy()
        cv2.rectangle(overlay, (6, 6), (self.width - 6, 48), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.42, frame, 0.58, 0)
        cv2.putText(frame, label, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            (14, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return self.raw_stdin.write(frame.tobytes())

    def close(self):
        return self.raw_stdin.close()

    def flush(self):
        return self.raw_stdin.flush()


class AnnotatedFFmpeg:
    def __init__(self, proc, width: int, height: int, label_state: dict[str, str]):
        self.proc = proc
        self.stdin = AnnotatedStdin(proc.stdin, width, height, label_state)

    def wait(self):
        return self.proc.wait()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", ""}:
        return False
    raise ValueError(f"cannot parse bool from {value!r}")


def parse_override_pairs(pairs: list[str] | None) -> dict[str, Any]:
    override_dict: dict[str, Any] = {}
    if not pairs:
        return override_dict
    if len(pairs) % 2 != 0:
        raise ValueError("overrides must be provided as --key value pairs")
    for i in range(0, len(pairs), 2):
        key = pairs[i].lstrip("-")
        value = pairs[i + 1]
        try:
            value = eval(value)
        except Exception:
            pass
        override_dict[key] = value
    return override_dict


def get_official_total_step_limit(spec: TaskSpec) -> int:
    step_limits = load_yaml(ROBOTWIN_ROOT / "task_config" / "_eval_step_limit.yml")
    return int(step_limits.get(spec.base_task_name, 1000))


def resolve_phase_step_limit(value: Any, total_step_limit: int) -> int:
    if value in (None, "", "None"):
        return int(total_step_limit)
    return int(value)


def get_env_class(spec: TaskSpec):
    module = importlib.import_module(spec.env_module)
    return getattr(module, spec.env_class)


def build_env_args(
    spec: TaskSpec,
    task_config: str,
    ckpt_setting: str,
    out_dir: Path,
    *,
    enable_eval_video_log: bool,
) -> dict[str, Any]:
    from envs import CONFIGS_PATH

    args = load_yaml(ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml")
    args.update(
        {
            "task_name": spec.base_task_name,
            "task_config": task_config,
            "ckpt_setting": ckpt_setting,
            "policy_name": "ACT",
            "eval_mode": True,
            "render_freq": 0,
            "eval_video_log": bool(enable_eval_video_log),
            "eval_video_save_dir": str(out_dir),
        }
    )

    embodiment_type = args.get("embodiment")
    embodiment_types = load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")

    def get_embodiment_file(embodiment: str) -> str:
        robot_file = embodiment_types[embodiment]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {embodiment}")
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
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = load_yaml(Path(args["left_robot_file"]) / "config.yml")
    args["right_embodiment_config"] = load_yaml(Path(args["right_robot_file"]) / "config.yml")

    camera_config = load_yaml(Path(CONFIGS_PATH) / "_camera_config.yml")
    head_camera_type = args["camera"]["head_camera_type"]
    wrist_camera_type = args["camera"]["wrist_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]
    args["wrist_camera_h"] = camera_config[wrist_camera_type]["h"]
    args["wrist_camera_w"] = camera_config[wrist_camera_type]["w"]
    return args


def resolve_subtask_checkpoint_dir(subtask: SubtaskSpec, usr_args: dict[str, Any]) -> str:
    explicit_key = f"subtask{subtask.subtask_id}_ckpt_dir"
    explicit_dir = usr_args.get(explicit_key)
    if explicit_dir not in (None, "", "None"):
        return str(explicit_dir)

    expert_data_num = usr_args.get("expert_data_num")
    if expert_data_num not in (None, "", "None"):
        ckpt_root = Path(str(usr_args.get("ckpt_root") or DEFAULT_ACT_CKPT_ROOT))
        ckpt_setting = str(usr_args["ckpt_setting"])
        return str(ckpt_root / f"act-{subtask.policy_task_name}" / f"{ckpt_setting}-{int(expert_data_num)}")

    return subtask.checkpoint_dir


def build_act_args(
    subtask: SubtaskSpec,
    checkpoint_dir: str,
    task_config: str,
    env_args: dict[str, Any],
    usr_args: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    args = load_yaml(ROBOTWIN_ROOT / "policy" / "ACT" / "deploy_policy.yml")
    args.update(
        {
            "task_name": subtask.policy_task_name,
            "task_config": task_config,
            "ckpt_setting": usr_args["ckpt_setting"],
            "policy_name": usr_args["policy_name"],
            "seed": int(seed),
            "instruction_type": usr_args["instruction_type"],
            "ckpt_dir": checkpoint_dir,
            "temporal_agg": parse_bool(usr_args["temporal_agg"]),
            "device": usr_args["device"],
            "left_arm_dim": len(env_args["left_embodiment_config"]["arm_joints_name"][0]),
            "right_arm_dim": len(env_args["right_embodiment_config"]["arm_joints_name"][1]),
        }
    )
    return args


def always_false(_self) -> bool:
    return False


def official_full_success(env) -> bool:
    try:
        final_check = getattr(env, "_check_subtask2_success", None)
        if callable(final_check):
            return bool(final_check())
        return bool(type(env).check_success(env))
    except Exception:
        return False


def expert_seed_passed(env_cls, env_args: dict[str, Any], episode_id: int, seed: int) -> bool:
    env = env_cls()
    try:
        env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **env_args)
        env.play_once()
        return bool(getattr(env, "plan_success", False) and type(env).check_success(env))
    except Exception:
        return False
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def setup_video(env, env_args: dict[str, Any], video_path: Path, fps: int, label_state: dict[str, str]):
    video_size = f"{env_args['head_camera_w']}x{env_args['head_camera_h']}"
    proc = subprocess.Popen(
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
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    wrapped = AnnotatedFFmpeg(proc, int(env_args["head_camera_w"]), int(env_args["head_camera_h"]), label_state)
    env._set_eval_video_ffmpeg(wrapped)
    return wrapped


def write_pause(env, seconds: int, fps: int, label_state: dict[str, str], text: str) -> None:
    old_label = label_state["label"]
    label_state["label"] = text
    for _ in range(int(seconds * fps)):
        obs = env.get_obs()
        frame = obs["observation"]["head_camera"]["rgb"]
        env.eval_video_ffmpeg.stdin.write(frame.tobytes())
    label_state["label"] = old_label


def decision_action(decision: dict[str, Any] | None) -> str:
    if not decision:
        return "none"
    command = decision.get("controller_command") or {}
    action = command.get("action")
    if action:
        return str(action)
    decision_name = decision.get("decision")
    if decision_name == "switch_policy":
        return "load_policy"
    if decision_name == "finish_task":
        return "finish"
    if decision_name == "abort_or_replan":
        return "abort"
    return "keep_policy"


def maybe_find_valid_seed(env_cls, env_args: dict[str, Any], start_seed: int, max_tries: int) -> int:
    if max_tries <= 0:
        return start_seed
    seed = start_seed
    for _ in range(max_tries):
        env = env_cls()
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **env_args)
            env.play_once()
            ok = bool(getattr(env, "plan_success", False) and type(env).check_success(env))
            env.close_env()
            if ok:
                return seed
        except Exception:
            try:
                env.close_env()
            except Exception:
                pass
        seed += 1
    raise RuntimeError(f"no valid expert seed found in {max_tries} tries from {start_seed}")


def run_phase_one(
    *,
    env,
    model,
    policy_model,
    bridge: DecomposedOpenClawBridge | None,
    llm_judge: EpisodeLLMJudge | None,
    spec: TaskSpec,
    subtask: SubtaskSpec,
    checkpoint_dir: str,
    task_config: str,
    phase_step_limit: int,
    total_step_limit: int,
    label_state: dict[str, str],
    decision_backend: str,
    mirror_every: int,
    decision_timeout: float,
    auto_switch_after_step: int | None,
) -> dict[str, Any]:
    env.task_name = spec.task_name
    env.step_lim = total_step_limit
    env.eval_success = False
    env.check_success = types.MethodType(always_false, env)
    env.set_instruction(subtask.instruction)
    gate_label = "OpenClaw visual gate" if decision_backend == "bridge" else "direct LLM visual gate"
    label_state["label"] = f"TASK1 {subtask.name}: {gate_label}"
    policy_model.reset_model(model)

    start = time.time()
    phase_start_take_action_cnt = int(env.take_action_cnt)
    phase_deadline = min(phase_start_take_action_cnt + int(phase_step_limit), int(total_step_limit))
    mirrored_payloads: list[str] = []
    switch_decision = None
    runtime_error = None
    switch_key = "switch_by_openclaw" if decision_backend == "bridge" else "switch_by_llm"

    while env.take_action_cnt < phase_deadline and env.take_action_cnt < env.step_lim:
        try:
            observation = env.get_obs()
            obs_for_act = policy_model.encode_obs(observation)
            if should_visual_gate(env.take_action_cnt, mirror_every):
                payload_id = build_payload_id(
                    spec.task_name,
                    subtask.subtask_id,
                    env.take_action_cnt,
                    len(mirrored_payloads),
                )
                mirrored_payloads.append(payload_id)
                if decision_backend == "bridge":
                    if bridge is None:
                        raise RuntimeError("decision_backend=bridge but bridge is not initialized")
                    payload_id = bridge.export_observation(
                        task_env=env,
                        model=model,
                        observation=observation,
                        obs_for_act=obs_for_act,
                        task_name=spec.task_name,
                        task_config=task_config,
                        phase_name="TASK1",
                        subtask_id=subtask.subtask_id,
                        subtask_count=len(spec.subtasks),
                        policy_task_name=subtask.policy_task_name,
                        checkpoint_dir=checkpoint_dir,
                        instruction_text=env.get_instruction(),
                        success_condition_text=subtask.success_condition,
                        visual_evidence=list(subtask.visual_evidence),
                        official_full_task_success=None,
                    )
                    mirrored_payloads[-1] = payload_id
                    decision = (
                        bridge.wait_for_decision(payload_id, decision_timeout)
                        if decision_timeout > 0
                        else bridge.poll_decision(payload_id)
                    )
                else:
                    if llm_judge is None:
                        raise RuntimeError("decision_backend=direct_llm but llm judge is not initialized")
                    decision = llm_judge.judge(
                        payload_id=payload_id,
                        phase_name="TASK1",
                        subtask=subtask,
                        step_count=env.take_action_cnt,
                        observation=observation,
                    )
                if decision_action(decision) == "load_policy":
                    switch_decision = decision
                    break
                if decision_action(decision) == "abort":
                    switch_decision = decision
                    runtime_error = decision.get("reason", "OpenClaw requested abort")
                    break
            if auto_switch_after_step is not None and env.take_action_cnt >= auto_switch_after_step:
                switch_decision = {
                    "decision": "switch_policy",
                    "controller_command": {"action": "load_policy"},
                    "reason": f"auto switch after step {auto_switch_after_step}",
                }
                break
            policy_model.eval(env, model, observation)
            if auto_switch_after_step is not None and env.take_action_cnt >= auto_switch_after_step:
                switch_decision = {
                    "decision": "switch_policy",
                    "controller_command": {"action": "load_policy"},
                    "reason": f"auto switch after step {auto_switch_after_step}",
                }
                break
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            break

    result = {
        "phase": "TASK1",
        "subtask_id": subtask.subtask_id,
        "subtask_name": subtask.name,
        "policy_task_name": subtask.policy_task_name,
        "checkpoint_dir": checkpoint_dir,
        "phase_start_take_action_cnt": phase_start_take_action_cnt,
        "phase_step_budget": int(phase_step_limit),
        "phase_deadline": int(phase_deadline),
        "official_total_step_limit": int(total_step_limit),
        "decision_backend": decision_backend,
        "switch_by_openclaw": bool(switch_decision and decision_action(switch_decision) == "load_policy")
        if decision_backend == "bridge"
        else False,
        "switch_by_llm": bool(switch_decision and decision_action(switch_decision) == "load_policy")
        if decision_backend != "bridge"
        else False,
        "switch_decision": switch_decision,
        "take_action_cnt": int(env.take_action_cnt),
        "step_limit": int(env.step_lim),
        "mirrored_payload_count": len(mirrored_payloads),
        "last_payload_id": mirrored_payloads[-1] if mirrored_payloads else None,
        "runtime_error": runtime_error,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    result[switch_key] = bool(switch_decision and decision_action(switch_decision) == "load_policy")
    return result


def run_phase_two(
    *,
    env,
    model,
    policy_model,
    bridge: DecomposedOpenClawBridge | None,
    spec: TaskSpec,
    subtask: SubtaskSpec,
    checkpoint_dir: str,
    task_config: str,
    phase_step_limit: int,
    total_step_limit: int,
    label_state: dict[str, str],
    mirror_every: int,
    mirror_observations: bool,
) -> dict[str, Any]:
    env.task_name = spec.task_name
    env.step_lim = total_step_limit
    env.eval_success = False
    env.check_success = types.MethodType(lambda self: official_full_success(self), env)
    env.set_instruction(subtask.instruction)
    label_state["label"] = f"TASK2 {subtask.name}: official full success"
    policy_model.reset_model(model)

    start = time.time()
    phase_start_take_action_cnt = int(env.take_action_cnt)
    phase_deadline = min(phase_start_take_action_cnt + int(phase_step_limit), int(total_step_limit))
    mirrored_payloads: list[str] = []
    runtime_error = None

    while env.take_action_cnt < phase_deadline and env.take_action_cnt < env.step_lim:
        try:
            observation = env.get_obs()
            obs_for_act = policy_model.encode_obs(observation)
            official_success = official_full_success(env)
            if mirror_observations and bridge is not None and should_visual_gate(env.take_action_cnt, mirror_every):
                payload_id = bridge.export_observation(
                    task_env=env,
                    model=model,
                    observation=observation,
                    obs_for_act=obs_for_act,
                    task_name=spec.task_name,
                    task_config=task_config,
                    phase_name="TASK2",
                    subtask_id=subtask.subtask_id,
                    subtask_count=len(spec.subtasks),
                    policy_task_name=subtask.policy_task_name,
                    checkpoint_dir=checkpoint_dir,
                    instruction_text=env.get_instruction(),
                    success_condition_text=subtask.success_condition,
                    visual_evidence=list(subtask.visual_evidence),
                    official_full_task_success=official_success,
                )
                mirrored_payloads.append(payload_id)
            if official_success:
                env.eval_success = True
                break
            policy_model.eval(env, model, observation)
            if env.eval_success:
                break
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            break

    final_success = official_full_success(env)
    return {
        "phase": "TASK2",
        "subtask_id": subtask.subtask_id,
        "subtask_name": subtask.name,
        "policy_task_name": subtask.policy_task_name,
        "checkpoint_dir": checkpoint_dir,
        "phase_start_take_action_cnt": phase_start_take_action_cnt,
        "phase_step_budget": int(phase_step_limit),
        "phase_deadline": int(phase_deadline),
        "official_total_step_limit": int(total_step_limit),
        "official_full_task_success": bool(final_success or env.eval_success),
        "take_action_cnt": int(env.take_action_cnt),
        "step_limit": int(env.step_lim),
        "mirrored_payload_count": len(mirrored_payloads),
        "last_payload_id": mirrored_payloads[-1] if mirrored_payloads else None,
        "runtime_error": runtime_error,
        "elapsed_seconds": round(time.time() - start, 3),
    }


def parse_direct_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", "--task_name", dest="task_name", choices=sorted(TASKS), required=True)
    parser.add_argument("--task-config", "--task_config", dest="task_config", default=None)
    parser.add_argument("--ckpt-setting", "--ckpt_setting", dest="ckpt_setting", default=None)
    parser.add_argument("--ckpt-root", "--ckpt_root", dest="ckpt_root", default=str(DEFAULT_ACT_CKPT_ROOT))
    parser.add_argument("--expert-data-num", "--expert_data_num", dest="expert_data_num", type=int, default=None)
    parser.add_argument("--subtask1-ckpt-dir", "--subtask1_ckpt_dir", dest="subtask1_ckpt_dir", default=None)
    parser.add_argument("--subtask2-ckpt-dir", "--subtask2_ckpt_dir", dest="subtask2_ckpt_dir", default=None)
    parser.add_argument("--policy-name", "--policy_name", dest="policy_name", default="ACT")
    parser.add_argument("--instruction-type", "--instruction_type", dest="instruction_type", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-num", "--test_num", dest="test_num", type=int, default=1)
    parser.add_argument("--temporal-agg", "--temporal_agg", dest="temporal_agg", default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expert-check", "--expert_check", dest="expert_check", action="store_true")
    parser.add_argument("--no-expert-check", "--no_expert_check", dest="expert_check", action="store_false")
    parser.set_defaults(expert_check=None)
    parser.add_argument("--expert-seed-tries", "--expert_seed_tries", dest="expert_seed_tries", type=int, default=0)
    parser.add_argument("--openclaw-mirror-root", "--openclaw_mirror_root", dest="openclaw_mirror_root", default="/data4/s2wxy/RoboTwin/results/openclaw_bridge")
    parser.add_argument("--openclaw-mirror-every", "--openclaw_mirror_every", dest="openclaw_mirror_every", type=int, default=10)
    parser.add_argument("--openclaw-decision-timeout", "--openclaw_decision_timeout", dest="openclaw_decision_timeout", type=float, default=0.0)
    parser.add_argument("--openclaw-decision-poll", "--openclaw_decision_poll", dest="openclaw_decision_poll", type=float, default=1.0)
    parser.add_argument("--mirror-phase2-observations", "--mirror_phase2_observations", dest="mirror_phase2_observations", action="store_true")
    parser.add_argument("--decision-backend", "--decision_backend", dest="decision_backend", choices=["direct_llm", "bridge"], default=DEFAULT_DECISION_BACKEND)
    parser.add_argument("--llm-provider", "--llm_provider", dest="llm_provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--llm-model", "--llm_model", dest="llm_model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-base-url", "--llm_base_url", dest="llm_base_url", default=None)
    parser.add_argument("--llm-api-key", "--llm_api_key", dest="llm_api_key", default=None)
    parser.add_argument("--llm-api-key-env", "--llm_api_key_env", dest="llm_api_key_env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--llm-openclaw-config", "--llm_openclaw_config", dest="llm_openclaw_config", default=DEFAULT_LLM_OPENCLAW_CONFIG)
    parser.add_argument("--llm-timeout-seconds", "--llm_timeout_seconds", dest="llm_timeout_seconds", type=float, default=DEFAULT_LLM_TIMEOUT_SECONDS)
    parser.add_argument("--llm-image-max-edge", "--llm_image_max_edge", dest="llm_image_max_edge", type=int, default=DEFAULT_LLM_IMAGE_MAX_EDGE)
    parser.add_argument("--llm-image-jpeg-quality", "--llm_image_jpeg_quality", dest="llm_image_jpeg_quality", type=int, default=DEFAULT_LLM_IMAGE_JPEG_QUALITY)
    parser.add_argument("--phase1-step-limit", "--phase1_step_limit", dest="phase1_step_limit", type=int, default=None)
    parser.add_argument("--phase2-step-limit", "--phase2_step_limit", dest="phase2_step_limit", type=int, default=None)
    parser.add_argument("--pause-seconds", "--pause_seconds", dest="pause_seconds", type=int, default=3)
    parser.add_argument("--video-fps", "--video_fps", dest="video_fps", type=int, default=10)
    parser.add_argument("--disable-eval-video-log", "--disable_eval_video_log", dest="disable_eval_video_log", action="store_true")
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default=None)
    parser.add_argument("--auto-switch-after-step", "--auto_switch_after_step", dest="auto_switch_after_step", type=int, default=None)
    return parser.parse_args(argv)


def parse_args_and_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.update(parse_override_pairs(args.overrides))
    return config


def load_user_args(argv: list[str] | None = None) -> tuple[dict[str, Any], str]:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--config" in argv:
        return parse_args_and_config(argv), "config"
    return vars(parse_direct_args(argv)), "cli"


def normalize_user_args(raw_args: dict[str, Any], mode: str) -> dict[str, Any]:
    usr_args = dict(raw_args)
    task_name = usr_args.get("task_name")
    if not task_name:
        raise ValueError("task_name is required")
    if task_name not in TASKS:
        raise ValueError(f"unsupported task_name: {task_name}")

    spec = TASKS[task_name]
    usr_args["policy_name"] = str(usr_args.get("policy_name") or "ACT")
    usr_args["task_config"] = str(usr_args.get("task_config") or spec.task_config)
    usr_args["ckpt_setting"] = str(usr_args.get("ckpt_setting") or usr_args["task_config"])
    usr_args["instruction_type"] = str(usr_args.get("instruction_type") or "unseen")
    usr_args["seed"] = int(usr_args.get("seed", 0))
    usr_args["test_num"] = int(usr_args.get("test_num", 100 if mode == "config" else 1))
    usr_args["device"] = str(usr_args.get("device") or "cuda:0")
    usr_args["ckpt_root"] = str(usr_args.get("ckpt_root") or DEFAULT_ACT_CKPT_ROOT)
    usr_args["temporal_agg"] = parse_bool(usr_args.get("temporal_agg", True))

    default_expert_check = True if mode == "config" else False
    expert_check_raw = usr_args.get("expert_check")
    usr_args["expert_check"] = default_expert_check if expert_check_raw is None else parse_bool(expert_check_raw)

    usr_args["expert_seed_tries"] = int(usr_args.get("expert_seed_tries", 0) or 0)
    usr_args["openclaw_mirror_root"] = str(
        usr_args.get("openclaw_mirror_root") or "/data4/s2wxy/RoboTwin/results/openclaw_bridge"
    )
    usr_args["openclaw_mirror_every"] = int(usr_args.get("openclaw_mirror_every", 10))
    usr_args["openclaw_decision_timeout"] = float(usr_args.get("openclaw_decision_timeout", 0.0))
    usr_args["openclaw_decision_poll"] = float(usr_args.get("openclaw_decision_poll", 1.0))
    usr_args["mirror_phase2_observations"] = parse_bool(usr_args.get("mirror_phase2_observations", False))
    usr_args["decision_backend"] = str(usr_args.get("decision_backend") or DEFAULT_DECISION_BACKEND)
    usr_args["llm_provider"] = str(usr_args.get("llm_provider") or DEFAULT_LLM_PROVIDER)
    usr_args["llm_model"] = str(usr_args.get("llm_model") or DEFAULT_LLM_MODEL)
    usr_args["llm_base_url"] = usr_args.get("llm_base_url")
    usr_args["llm_api_key"] = usr_args.get("llm_api_key")
    usr_args["llm_api_key_env"] = str(usr_args.get("llm_api_key_env") or DEFAULT_LLM_API_KEY_ENV)
    usr_args["llm_openclaw_config"] = str(usr_args.get("llm_openclaw_config") or DEFAULT_LLM_OPENCLAW_CONFIG)
    usr_args["llm_timeout_seconds"] = float(usr_args.get("llm_timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS))
    usr_args["llm_image_max_edge"] = int(usr_args.get("llm_image_max_edge", DEFAULT_LLM_IMAGE_MAX_EDGE))
    usr_args["llm_image_jpeg_quality"] = int(usr_args.get("llm_image_jpeg_quality", DEFAULT_LLM_IMAGE_JPEG_QUALITY))
    usr_args["pause_seconds"] = int(usr_args.get("pause_seconds", 3))
    usr_args["video_fps"] = int(usr_args.get("video_fps", 10))
    usr_args["disable_eval_video_log"] = parse_bool(usr_args.get("disable_eval_video_log", False))

    if usr_args.get("phase1_step_limit") not in (None, "", "None"):
        usr_args["phase1_step_limit"] = int(usr_args["phase1_step_limit"])
    else:
        usr_args["phase1_step_limit"] = None
    if usr_args.get("phase2_step_limit") not in (None, "", "None"):
        usr_args["phase2_step_limit"] = int(usr_args["phase2_step_limit"])
    else:
        usr_args["phase2_step_limit"] = None
    if usr_args.get("expert_data_num") not in (None, "", "None"):
        usr_args["expert_data_num"] = int(usr_args["expert_data_num"])
    else:
        usr_args["expert_data_num"] = None
    if usr_args.get("auto_switch_after_step") not in (None, "", "None"):
        usr_args["auto_switch_after_step"] = int(usr_args["auto_switch_after_step"])
    else:
        usr_args["auto_switch_after_step"] = None

    return usr_args


def get_save_dir(usr_args: dict[str, Any]) -> Path:
    if usr_args.get("out_dir"):
        save_dir = Path(str(usr_args["out_dir"]))
    else:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = Path(
            f"eval_result/{usr_args['task_name']}/ACT_decomposed_openclaw/"
            f"{usr_args['task_config']}/{usr_args['ckpt_setting']}/{now}"
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def build_episode_dir(save_dir: Path, episode_id: int, seed: int, test_num: int) -> Path:
    if test_num == 1 and not list(save_dir.iterdir()):
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir
    episode_dir = save_dir / f"episode{episode_id:03d}_seed{seed}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    return episode_dir


def build_payload_id(task_name: str, subtask_id: int, step_count: int, mirror_index: int) -> str:
    return (
        f"robotwin-decomp-{task_name}-subtask{subtask_id}-"
        f"step{int(step_count):04d}-mirror{int(mirror_index):05d}"
    ).replace("/", "-")


def should_visual_gate(step_count: int, mirror_every: int) -> bool:
    return int(step_count) % max(1, int(mirror_every)) == 0


def encode_rgb_for_llm(image_rgb: np.ndarray, max_edge: int, jpeg_quality: int) -> tuple[bytes, tuple[int, int]]:
    image = image_rgb
    height, width = image.shape[:2]
    if max_edge > 0 and max(height, width) > max_edge:
        scale = float(max_edge) / float(max(height, width))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("failed to encode head camera image for llm")
    encoded = buf.tobytes()
    return encoded, (int(image.shape[1]), int(image.shape[0]))


def build_llm_system_prompt() -> str:
    return (
        "You are a strict RoboTwin visual gate for decomposed ACT tasks. "
        "Inspect only the attached head-camera image and the provided text fields. "
        "Return exactly one strict JSON object only: "
        '{"task1_status":"完成"} or {"task1_status":"未完成"}. '
        "If any success condition is not clearly visible in this single image, "
        "or the object may still be outside the target / held by the gripper / occluded, "
        'return {"task1_status":"未完成"}.'
    )


def build_llm_user_prompt(subtask: SubtaskSpec) -> str:
    lines = [
        "RoboTwin task-1 completion check.",
        "Do not call tools.",
        "Do not explain your reasoning.",
        "Return exactly one strict JSON object only.",
        "Allowed outputs:",
        '{"task1_status":"完成"}',
        "or",
        '{"task1_status":"未完成"}',
        "",
        f"task_text: {subtask.instruction}",
        f"success_condition_text: {subtask.success_condition}",
    ]
    if subtask.visual_evidence:
        lines.append("visual_evidence_required:")
        lines.extend(f"- {item}" for item in subtask.visual_evidence)
    return "\n".join(lines)


def response_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE | re.DOTALL).strip()
    for candidate in (stripped, fenced):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return obj

    start = fenced.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(fenced[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = fenced[start : index + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def normalize_gate_decision(
    raw_decision: dict[str, Any] | None,
    *,
    payload_id: str,
    source: str,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    normalized = dict(raw_decision or {})
    status = (
        normalized.get("task1_status")
        or normalized.get("任务一")
        or normalized.get("status")
        or normalized.get("result")
    )
    if isinstance(status, str):
        compact = status.strip().lower()
        if compact in {"完成", "done", "complete", "completed", "success", "true"}:
            normalized["task1_status"] = "完成"
            normalized["decision"] = "switch_policy"
            normalized["controller_command"] = {"action": "load_policy"}
        else:
            normalized["task1_status"] = "未完成"
            normalized["decision"] = "continue_current_policy"
            normalized["controller_command"] = {"action": "keep_policy"}
    else:
        normalized["task1_status"] = "未完成"
        normalized["decision"] = "abort_or_replan" if fallback_reason else "continue_current_policy"
        normalized["controller_command"] = {"action": "abort" if fallback_reason else "keep_policy"}

    normalized["payload_id"] = payload_id
    normalized["source"] = source
    normalized["created_at_ms"] = int(time.time() * 1000)
    if fallback_reason:
        normalized["reason"] = fallback_reason
    return normalized


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_openclaw_provider_config(path: str, provider: str) -> dict[str, Any] | None:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    providers = (((data or {}).get("models") or {}).get("providers") or {})
    provider_data = providers.get(provider)
    return provider_data if isinstance(provider_data, dict) else None


def resolve_direct_llm_config(usr_args: dict[str, Any]) -> DirectLLMConfig:
    provider = str(usr_args.get("llm_provider") or DEFAULT_LLM_PROVIDER)
    model = str(usr_args.get("llm_model") or DEFAULT_LLM_MODEL)
    api_key = usr_args.get("llm_api_key")
    base_url = usr_args.get("llm_base_url")
    config_path = str(usr_args.get("llm_openclaw_config") or DEFAULT_LLM_OPENCLAW_CONFIG)
    provider_config = load_openclaw_provider_config(config_path, provider)
    if not api_key:
        env_name = str(usr_args.get("llm_api_key_env") or DEFAULT_LLM_API_KEY_ENV)
        api_key = os.environ.get(env_name)
    if not api_key and provider_config:
        api_key = provider_config.get("apiKey")
    if not base_url and provider_config:
        base_url = provider_config.get("baseUrl")
    if not api_key:
        raise RuntimeError(
            "missing llm api key; set --llm_api_key / --llm-api-key, "
            f"export {usr_args.get('llm_api_key_env') or DEFAULT_LLM_API_KEY_ENV}, "
            "or provide an OpenClaw config that contains the provider apiKey"
        )
    if not base_url:
        raise RuntimeError(
            "missing llm base url; set --llm_base_url / --llm-base-url "
            "or provide an OpenClaw config that contains the provider baseUrl"
        )
    return DirectLLMConfig(
        provider=provider,
        model=model,
        base_url=str(base_url).rstrip("/"),
        api_key=str(api_key),
        timeout_seconds=float(usr_args.get("llm_timeout_seconds") or DEFAULT_LLM_TIMEOUT_SECONDS),
        image_max_edge=int(usr_args.get("llm_image_max_edge") or DEFAULT_LLM_IMAGE_MAX_EDGE),
        image_jpeg_quality=int(usr_args.get("llm_image_jpeg_quality") or DEFAULT_LLM_IMAGE_JPEG_QUALITY),
        max_tokens=DEFAULT_LLM_MAX_TOKENS,
        retry_max_tokens=DEFAULT_LLM_RETRY_MAX_TOKENS,
        reasoning_effort=DEFAULT_LLM_REASONING_EFFORT,
    )


class EpisodeLLMJudge:
    def __init__(self, *, config: DirectLLMConfig, episode_dir: Path, run_id: str):
        self.config = config
        self.run_id = run_id
        self.episode_dir = episode_dir
        self.log_path = episode_dir / "llm_conversation.jsonl"
        self.call_dir = episode_dir / "llm_calls"
        self.image_dir = episode_dir / "llm_images"
        self.call_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def judge(
        self,
        *,
        payload_id: str,
        phase_name: str,
        subtask: SubtaskSpec,
        step_count: int,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        prompt_system = build_llm_system_prompt()
        prompt_user = build_llm_user_prompt(subtask)
        image_bytes, image_size = encode_rgb_for_llm(
            observation["observation"]["head_camera"]["rgb"],
            self.config.image_max_edge,
            self.config.image_jpeg_quality,
        )
        image_path = self.image_dir / f"{payload_id}_head.jpg"
        image_path.write_bytes(image_bytes)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        raw_response_text = ""
        provider_response_json: dict[str, Any] | None = None
        response_status_code: int | None = None
        elapsed_seconds = 0.0
        decision: dict[str, Any]
        error_text = None
        attempts: list[dict[str, Any]] = []
        start = time.time()
        try:
            raw_decision = None
            last_error: RuntimeError | None = None
            max_tokens_plan: list[int] = []
            for value in (self.config.max_tokens, self.config.retry_max_tokens):
                if value > 0 and value not in max_tokens_plan:
                    max_tokens_plan.append(int(value))

            for attempt_index, max_tokens in enumerate(max_tokens_plan, start=1):
                request_payload = {
                    "model": self.config.model,
                    "temperature": 0,
                    "max_tokens": int(max_tokens),
                    "reasoning_effort": self.config.reasoning_effort,
                    "messages": [
                        {"role": "system", "content": prompt_system},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_user},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                            ],
                        },
                    ],
                }
                attempt_start = time.time()
                response = requests.post(
                    f"{self.config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=self.config.timeout_seconds,
                )
                response_status_code = response.status_code
                raw_response_text = response.text
                if response.status_code >= 400:
                    raise RuntimeError(f"llm http {response.status_code}: {response.text[:2000]}")
                provider_response_json = response.json()
                choice = ((provider_response_json.get("choices") or [{}])[0]) or {}
                finish_reason = choice.get("finish_reason")
                content = response_content_to_text(((choice.get("message") or {}).get("content")))
                raw_decision = extract_json_from_text(content)
                attempt_error = None
                if raw_decision is None:
                    attempt_error = f"could not extract strict json from llm response: {content}"
                    last_error = RuntimeError(attempt_error)
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "max_tokens": int(max_tokens),
                        "http_status": response_status_code,
                        "finish_reason": finish_reason,
                        "content": content,
                        "error": attempt_error,
                        "elapsed_seconds": round(time.time() - attempt_start, 3),
                    }
                )
                if raw_decision is not None:
                    break
                if finish_reason == "length" and attempt_index < len(max_tokens_plan):
                    continue
                raise last_error or RuntimeError("could not extract strict json from llm response")

            elapsed_seconds = round(time.time() - start, 3)
            decision = normalize_gate_decision(
                raw_decision,
                payload_id=payload_id,
                source=f"direct_llm:{self.config.provider}/{self.config.model}",
            )
        except Exception as exc:
            elapsed_seconds = round(time.time() - start, 3)
            error_text = f"{type(exc).__name__}: {exc}"
            # Keep the current policy instead of poisoning the whole episode on one bad LLM reply.
            decision = normalize_gate_decision(
                {"task1_status": "未完成"},
                payload_id=payload_id,
                source=f"direct_llm:{self.config.provider}/{self.config.model}",
            )
            decision["reason"] = error_text

        log_record = {
            "run_id": self.run_id,
            "payload_id": payload_id,
            "phase_name": phase_name,
            "step_count": int(step_count),
            "created_at": datetime.now().isoformat(),
            "provider": self.config.provider,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "image_path": str(image_path),
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "prompt": {
                "system": prompt_system,
                "user": prompt_user,
            },
            "response": {
                "http_status": response_status_code,
                "raw_text": raw_response_text,
                "provider_json": provider_response_json,
                "error": error_text,
                "elapsed_seconds": elapsed_seconds,
                "attempts": attempts,
            },
            "decision": decision,
        }
        append_jsonl(self.log_path, log_record)
        (self.call_dir / f"{payload_id}.json").write_text(
            json.dumps(log_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return decision


def run_single_episode(
    *,
    spec: TaskSpec,
    usr_args: dict[str, Any],
    episode_id: int,
    setup_seed: int,
    episode_dir: Path,
    clear_cache: bool,
) -> dict[str, Any]:
    task_config = usr_args["task_config"]
    official_total_step_limit = get_official_total_step_limit(spec)
    phase1_step_limit = resolve_phase_step_limit(usr_args["phase1_step_limit"], official_total_step_limit)
    phase2_step_limit = resolve_phase_step_limit(usr_args["phase2_step_limit"], official_total_step_limit)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"decomp_{spec.task_name}_{run_stamp}_ep{episode_id:03d}_seed{setup_seed}"

    env_args = build_env_args(
        spec,
        task_config,
        usr_args["ckpt_setting"],
        episode_dir,
        enable_eval_video_log=not usr_args["disable_eval_video_log"],
    )
    env_cls = get_env_class(spec)
    env = env_cls()
    policy_model = importlib.import_module(usr_args["policy_name"])
    label_state = {"label": "initializing"}
    decision_backend = usr_args["decision_backend"]
    bridge = None
    if decision_backend == "bridge":
        bridge = DecomposedOpenClawBridge(
            root_dir=usr_args["openclaw_mirror_root"],
            run_id=run_id,
            mirror_every=usr_args["openclaw_mirror_every"],
            decision_poll_interval=usr_args["openclaw_decision_poll"],
        )
    llm_judge = None
    if decision_backend == "direct_llm":
        llm_judge = EpisodeLLMJudge(
            config=resolve_direct_llm_config(usr_args),
            episode_dir=episode_dir,
            run_id=run_id,
        )
    report_path = episode_dir / "report.json"
    video_path = episode_dir / f"episode{episode_id:03d}.mp4"
    llm_log_path = episode_dir / "llm_conversation.jsonl"
    subtask1, subtask2 = spec.subtasks
    subtask1_ckpt_dir = resolve_subtask_checkpoint_dir(subtask1, usr_args)
    subtask2_ckpt_dir = resolve_subtask_checkpoint_dir(subtask2, usr_args)

    print("===== RoboTwin decomposed ACT visual gate controller =====", flush=True)
    print(f"episode_id={episode_id}", flush=True)
    print(f"decision_backend={decision_backend}", flush=True)
    print(f"frequency_basis=visual_gate_1Hz_act_runs_continuously", flush=True)
    print(f"openclaw_mirror_every={usr_args['openclaw_mirror_every']}", flush=True)
    print(f"run_id={run_id}", flush=True)
    print(f"episode_dir={episode_dir}", flush=True)
    print(f"bridge_run_dir={bridge.run_dir if bridge is not None else None}", flush=True)
    print(f"llm_log_path={llm_log_path if llm_judge is not None else None}", flush=True)
    if llm_judge is not None:
        print(
            f"llm_provider={llm_judge.config.provider} llm_model={llm_judge.config.model} "
            f"llm_base_url={llm_judge.config.base_url}",
            flush=True,
        )
    print(f"setup_seed={setup_seed}", flush=True)
    print(f"official_total_step_limit={official_total_step_limit}", flush=True)
    print("default_phase_step_limits=shared_official_total_budget", flush=True)
    print(f"resolved_phase_step_limits={[phase1_step_limit, phase2_step_limit]}", flush=True)

    phase1 = None
    phase2 = None
    status = "error"
    try:
        env.setup_demo(now_ep_num=episode_id, seed=setup_seed, is_test=True, **env_args)
        env.step_lim = official_total_step_limit
        if not usr_args["disable_eval_video_log"]:
            setup_video(env, env_args, video_path, usr_args["video_fps"], label_state)

        print(f"LOADING_TASK1_MODEL {subtask1.policy_task_name}", flush=True)
        model1 = policy_model.get_model(build_act_args(subtask1, subtask1_ckpt_dir, task_config, env_args, usr_args, usr_args["seed"]))
        print("LOADED_TASK1_MODEL " + json.dumps({"ckpt_dir": subtask1_ckpt_dir}), flush=True)

        phase1 = run_phase_one(
            env=env,
            model=model1,
            policy_model=policy_model,
            bridge=bridge,
            llm_judge=llm_judge,
            spec=spec,
            subtask=subtask1,
            checkpoint_dir=subtask1_ckpt_dir,
            task_config=task_config,
            phase_step_limit=phase1_step_limit,
            total_step_limit=official_total_step_limit,
            label_state=label_state,
            decision_backend=decision_backend,
            mirror_every=usr_args["openclaw_mirror_every"],
            decision_timeout=usr_args["openclaw_decision_timeout"],
            auto_switch_after_step=usr_args["auto_switch_after_step"],
        )
        print("TASK1_RESULT " + json.dumps(phase1, ensure_ascii=False), flush=True)

        if decision_action(phase1.get("switch_decision")) == "load_policy" and not phase1["runtime_error"]:
            status = "running_task2"
            if not usr_args["disable_eval_video_log"]:
                write_pause(
                    env,
                    usr_args["pause_seconds"],
                    usr_args["video_fps"],
                    label_state,
                    f"PAUSE {usr_args['pause_seconds']}s: visual gate switch to TASK2",
                )

            print("UNLOADING_TASK1_MODEL", flush=True)
            del model1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"LOADING_TASK2_MODEL {subtask2.policy_task_name}", flush=True)
            model2 = policy_model.get_model(build_act_args(subtask2, subtask2_ckpt_dir, task_config, env_args, usr_args, usr_args["seed"]))
            print("LOADED_TASK2_MODEL " + json.dumps({"ckpt_dir": subtask2_ckpt_dir}), flush=True)

            phase2 = run_phase_two(
                env=env,
                model=model2,
                policy_model=policy_model,
                bridge=bridge,
                spec=spec,
                subtask=subtask2,
                checkpoint_dir=subtask2_ckpt_dir,
                task_config=task_config,
                phase_step_limit=phase2_step_limit,
                total_step_limit=official_total_step_limit,
                label_state=label_state,
                mirror_every=usr_args["openclaw_mirror_every"],
                mirror_observations=usr_args["mirror_phase2_observations"],
            )
            print("TASK2_RESULT " + json.dumps(phase2, ensure_ascii=False), flush=True)
            status = "completed"
        else:
            status = "stopped_before_task2"
    finally:
        try:
            if getattr(env, "eval_video_ffmpeg", None):
                env._del_eval_video_ffmpeg()
        except Exception:
            pass
        try:
            env.close_env(clear_cache=clear_cache)
        except Exception:
            pass

    success = bool(phase2 and phase2.get("official_full_task_success") and not phase2.get("runtime_error"))
    report = {
        "status": status,
        "episode_id": episode_id,
        "seed": setup_seed,
        "next_seed": setup_seed + 1,
        "success": success,
        "task_name": spec.task_name,
        "task_config": task_config,
        "ckpt_setting": usr_args["ckpt_setting"],
        "instruction_type": usr_args["instruction_type"],
        "policy_name": usr_args["policy_name"],
        "same_env_second_phase": True,
        "decision_backend": decision_backend,
        "step_budget": {
            "official_total_step_limit": official_total_step_limit,
            "subtask_count": len(spec.subtasks),
            "default_phase_step_limits": "shared_official_total_budget",
            "resolved_phase_step_limits": [phase1_step_limit, phase2_step_limit],
            "budget_rule": (
                "By default each phase may run until the shared official total "
                "env.step_lim is reached; explicit phase limits only add tighter "
                "per-phase deadlines."
            ),
        },
        "checkpoint_dirs": {
            "subtask1": subtask1_ckpt_dir,
            "subtask2": subtask2_ckpt_dir,
        },
        "frequency_decision": {
            "gate_layer_hz": 1,
            "decision_backend": decision_backend,
            "video_fps": usr_args["video_fps"],
            "openclaw_mirror_every": usr_args["openclaw_mirror_every"],
            "reason": "The visual gate judges whether subtask 1 is complete. RoboTwin video is 10fps, so every 10 frames gives a 1Hz visual gate.",
        },
        "phase1": phase1,
        "phase2": phase2,
        "paths": {
            "episode_dir": str(episode_dir),
            "video": str(video_path),
            "report": str(report_path),
            "bridge_run_dir": str(bridge.run_dir) if bridge is not None else None,
            "llm_conversation_log": str(llm_log_path) if llm_judge is not None else None,
            "llm_call_dir": str(llm_judge.call_dir) if llm_judge is not None else None,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FINAL_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)
    return report


def write_batch_summary(save_dir: Path, usr_args: dict[str, Any], episode_records: list[dict[str, Any]]) -> None:
    success_num = sum(1 for record in episode_records if record.get("success"))
    test_num = max(len(episode_records), 1)
    result_path = save_dir / "_result.txt"
    detail_path = save_dir / "episode_records.json"

    lines = [
        f"Task Name: {usr_args['task_name']}",
        f"Policy Name: {usr_args['policy_name']}_decomposed_openclaw",
        f"Decision Backend: {usr_args['decision_backend']}",
        f"Task Config: {usr_args['task_config']}",
        f"Ckpt Setting: {usr_args['ckpt_setting']}",
        f"Instruction Type: {usr_args['instruction_type']}",
        f"Test Num: {len(episode_records)}",
        f"Success: {success_num}/{len(episode_records)}",
        f"Success Rate: {round(success_num / test_num * 100, 1)}%",
    ]
    if episode_records:
        last_record = episode_records[-1]
        checkpoint_dirs = last_record.get("checkpoint_dirs", {})
        lines.append(f"Subtask1 Ckpt: {checkpoint_dirs.get('subtask1')}")
        lines.append(f"Subtask2 Ckpt: {checkpoint_dirs.get('subtask2')}")

    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    detail_path.write_text(json.dumps(episode_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Data has been saved to {result_path}")
    print(f"Episode details have been saved to {detail_path}")


def main() -> None:
    os.chdir(ROBOTWIN_ROOT)
    sys.path.append("./")
    sys.path.append("./policy")
    sys.path.append("./description/utils")

    raw_args, mode = load_user_args()
    usr_args = normalize_user_args(raw_args, mode)
    spec = TASKS[usr_args["task_name"]]
    save_dir = get_save_dir(usr_args)
    env_cls = get_env_class(spec)
    expert_probe_env_args = build_env_args(
        spec,
        usr_args["task_config"],
        usr_args["ckpt_setting"],
        save_dir,
        enable_eval_video_log=False,
    )
    test_num = int(usr_args["test_num"])
    start_seed = 100000 * (1 + usr_args["seed"])
    clear_cache_freq = max(1, int(expert_probe_env_args.get("clear_cache_freq", 1)))

    print("============= Config =============\n")
    print(f"Task Name: {usr_args['task_name']}")
    print(f"Policy Name: {usr_args['policy_name']}_decomposed_openclaw")
    print(f"Decision Backend: {usr_args['decision_backend']}")
    print(f"Task Config: {usr_args['task_config']}")
    print(f"Ckpt Setting: {usr_args['ckpt_setting']}")
    print(f"Instruction Type: {usr_args['instruction_type']}")
    print(f"Test Num: {test_num}")
    print(f"Expert Check: {usr_args['expert_check']}")
    print(f"Save Dir: {save_dir}")
    print("\n==================================")

    episode_records: list[dict[str, Any]] = []
    policy_success_num = 0
    accepted_seed_num = 0
    now_seed = start_seed
    invalid_seed_num = 0

    while accepted_seed_num < test_num:
        if usr_args["expert_check"]:
            expert_ok = expert_seed_passed(env_cls, expert_probe_env_args, accepted_seed_num, now_seed)
            if not expert_ok:
                now_seed += 1
                invalid_seed_num += 1
                if usr_args["expert_seed_tries"] > 0 and invalid_seed_num >= usr_args["expert_seed_tries"]:
                    raise RuntimeError(
                        f"no valid expert seed found in {usr_args['expert_seed_tries']} tries from {start_seed}"
                    )
                continue
        invalid_seed_num = 0
        accepted_seed_num += 1
        episode_id = len(episode_records)
        episode_dir = build_episode_dir(save_dir, episode_id, now_seed, test_num)
        episode_record = run_single_episode(
            spec=spec,
            usr_args=usr_args,
            episode_id=episode_id,
            setup_seed=now_seed,
            episode_dir=episode_dir,
            clear_cache=((accepted_seed_num + 1) % clear_cache_freq == 0),
        )
        episode_records.append(episode_record)
        if episode_record["success"]:
            policy_success_num += 1

        print(
            f"\033[93m{usr_args['task_name']}\033[0m | "
            f"\033[94m{usr_args['policy_name']}_decomposed_openclaw\033[0m | "
            f"\033[92m{usr_args['task_config']}\033[0m | "
            f"\033[91m{usr_args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{policy_success_num}/{len(episode_records)}\033[0m => "
            f"\033[95m{round(policy_success_num / len(episode_records) * 100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n",
            flush=True,
        )
        now_seed += 1

    write_batch_summary(save_dir, usr_args, episode_records)


if __name__ == "__main__":
    main()
