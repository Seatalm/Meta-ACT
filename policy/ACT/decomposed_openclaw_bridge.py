#!/usr/bin/env python3
"""Bridge utilities for RoboTwin decomposed ACT -> OpenClaw phase decisions."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2


def _jpeg_base64_from_rgb(image):
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )
    if not ok:
        raise RuntimeError("failed to encode image as jpeg")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _resize_rgb(observation: dict[str, Any], camera_name: str):
    return cv2.resize(
        observation["observation"][camera_name]["rgb"],
        (384, 288),
        interpolation=cv2.INTER_LINEAR,
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


class DecomposedOpenClawBridge:
    """File-spool bridge.

    The controller writes observation payloads to pending/. A local OpenClaw ingest
    process can read them, ask OpenClaw, then write decisions/<payload_id>.json.
    """

    def __init__(
        self,
        *,
        root_dir: str | Path,
        run_id: str,
        mirror_every: int = 5,
        decision_poll_interval: float = 1.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.run_id = run_id
        self.run_dir = self.root_dir / run_id
        self.pending_dir = self.run_dir / "pending"
        self.processed_dir = self.run_dir / "processed"
        self.decisions_dir = self.run_dir / "decisions"
        self.consumed_decisions_dir = self.run_dir / "decisions_consumed"
        self.mirror_every = max(1, int(mirror_every))
        self.decision_poll_interval = float(decision_poll_interval)
        self.mirror_counter = 0
        for directory in (
            self.pending_dir,
            self.processed_dir,
            self.decisions_dir,
            self.consumed_decisions_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def should_mirror(self, step_count: int) -> bool:
        return int(step_count) % self.mirror_every == 0

    def export_observation(
        self,
        *,
        task_env,
        model,
        observation: dict[str, Any],
        obs_for_act: dict[str, Any],
        task_name: str,
        task_config: str,
        phase_name: str,
        subtask_id: int,
        subtask_count: int,
        policy_task_name: str,
        checkpoint_dir: str,
        instruction_text: str | None,
        success_condition_text: str,
        visual_evidence: list[str],
        official_full_task_success: bool | None = None,
    ) -> str:
        head_cam = _resize_rgb(observation, "head_camera")
        left_cam = _resize_rgb(observation, "left_camera")
        right_cam = _resize_rgb(observation, "right_camera")
        step_count = int(getattr(task_env, "take_action_cnt", 0))
        payload_id = (
            f"robotwin-decomp-{task_name}-subtask{subtask_id}-"
            f"step{step_count:04d}-mirror{self.mirror_counter:05d}"
        ).replace("/", "-")

        payload = {
            "payload_version": 2,
            "source": "robotwin_decomposed_act_controller",
            "controller_mode": "decomposed_act",
            "payload_id": payload_id,
            "created_at_ms": int(time.time() * 1000),
            "run_id": self.run_id,
            "task_name": task_name,
            "task_config": task_config,
            "phase_name": phase_name,
            "subtask_id": int(subtask_id),
            "subtask_count": int(subtask_count),
            "policy_task_name": policy_task_name,
            "checkpoint_dir": checkpoint_dir,
            "instruction_text": instruction_text,
            "success_condition_text": success_condition_text,
            "visual_evidence": visual_evidence,
            "take_action_cnt_before_eval": step_count,
            "step_limit": getattr(task_env, "step_lim", None),
            "policy_t": getattr(model, "t", None),
            "qpos_for_act": list(obs_for_act["qpos"]),
            "camera_order": ["head_cam", "left_cam", "right_cam"],
            "phase_context": {
                "switch_authority": (
                    "openclaw_visual_judgement"
                    if int(subtask_id) == 1
                    else "robotwin_official_full_task_check"
                ),
                "official_subtask_success_available": False,
                "official_full_task_success": official_full_task_success,
                "subtask_success_condition": success_condition_text,
                "visual_evidence_required": visual_evidence,
            },
            "expected_openclaw_response": {
                "strict_json": True,
                "allowed_decisions": [
                    "continue_current_policy",
                    "switch_policy",
                    "finish_task",
                    "abort_or_replan",
                ],
                "decision_file": str(self.decisions_dir / f"{payload_id}.json"),
            },
        }

        for key, image in (
            ("head_cam", head_cam),
            ("left_cam", left_cam),
            ("right_cam", right_cam),
        ):
            encoded = _jpeg_base64_from_rgb(image)
            payload[f"{key}_jpeg_base64"] = encoded
            payload[f"{key}_mime_type"] = "image/jpeg"
            payload[f"{key}_display_size"] = [384, 288]
            payload[f"{key}_sha256"] = hashlib.sha256(base64.b64decode(encoded)).hexdigest()

        payload["act_input_signature_sha256"] = hashlib.sha256(
            json.dumps(
                {
                    "task_name": payload["policy_task_name"],
                    "qpos_for_act": payload["qpos_for_act"],
                    "camera_hashes": [
                        payload["head_cam_sha256"],
                        payload["left_cam_sha256"],
                        payload["right_cam_sha256"],
                    ],
                    "subtask_id": payload["subtask_id"],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        _atomic_write_json(self.pending_dir / f"{payload_id}.json", payload)
        self.mirror_counter += 1
        return payload_id

    def poll_decision(self, payload_id: str) -> dict[str, Any] | None:
        path = self.decisions_dir / f"{payload_id}.json"
        if not path.exists():
            return None
        try:
            decision = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "decision": "abort_or_replan",
                "controller_command": {"action": "abort"},
                "reason": f"invalid decision json for {payload_id}: {exc}",
            }
        consumed_path = self.consumed_decisions_dir / path.name
        try:
            os.replace(path, consumed_path)
        except OSError:
            pass
        decision["_decision_file"] = str(consumed_path)
        return decision

    def wait_for_decision(self, payload_id: str, timeout_s: float) -> dict[str, Any] | None:
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() <= deadline:
            decision = self.poll_decision(payload_id)
            if decision is not None:
                return decision
            time.sleep(self.decision_poll_interval)
        return None
