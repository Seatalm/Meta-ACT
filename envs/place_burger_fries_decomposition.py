from .place_burger_fries import place_burger_fries
from .utils import ArmTag
import numpy as np


class place_burger_fries_decomposition(place_burger_fries):
    def __init__(self, **args):
        super().__init__(**args)
        self.current_subtask = 0
        self.subtask_done = [False, False]
        self.subtask_success = [False, False]

    def play_once(self):
        """兼容常规执行：顺序执行两个独立子任务。"""
        self._execute_subtask(1)
        if not self.subtask_success[0]:
            print("[FAIL] 子任务1失败：未能将汉堡放到托盘上")
            self.info["info"] = {
                "subtask1_success": False,
                "subtask2_success": False,
                "failed_subtask": 1,
                "failed_reason": "子任务1失败：未能将汉堡放到托盘上",
            }
            return self.info

        self._reset_after_subtask(subtask_id=1)

        self._execute_subtask(2)
        if not self.subtask_success[1]:
            print("[FAIL] 子任务2失败：未能将薯条放到托盘上")

        self.info["info"] = {
            "subtask1_success": self.subtask_success[0],
            "subtask2_success": self.subtask_success[1],
            "failed_subtask": None if (self.subtask_success[0] and self.subtask_success[1]) else 2,
            "{A}": f"006_hamburg/base{self.object1_id}",
            "{B}": f"008_tray/base{self.tray_id}",
            "{C}": f"005_french-fries/base{self.object2_id}",
        }
        return self.info

    def _execute_subtask(self, subtask_id):
        """环境内的独立执行逻辑。"""
        self.current_subtask = subtask_id
        self.subtask_done[subtask_id - 1] = False
        self.subtask_success[subtask_id - 1] = False

        self._setup_subtask_config(subtask_id)

        motion_plan_func = getattr(self, f"_subtask{subtask_id}_motion_plan")
        check_success_func = getattr(
            self, f"_check_subtask{subtask_id}_success")

        motion_plan_func()

        self.subtask_success[subtask_id - 1] = check_success_func()
        self.subtask_done[subtask_id - 1] = True

    def _setup_subtask_config(self, subtask_id):
        """每个子任务开始前清理上一段轨迹缓存。"""
        self.current_subtask = subtask_id
        self.plan_success = True
        self.FRAME_IDX = 0
        self._clear_path_cache()

    def _clear_path_cache(self):
        """清空上一段动作的轨迹数组。"""
        self.left_joint_path = []
        self.right_joint_path = []
        self.left_cnt = 0
        self.right_cnt = 0

    def _subtask1_motion_plan(self):
        """子任务1：左臂将汉堡放到托盘上。"""
        arm_tag_left = ArmTag("left")

        self.move(
            self.grasp_actor(
                self.hamburg,
                arm_tag=arm_tag_left,
                pre_grasp_dis=0.1,
            )
        )

        if not self.plan_success:
            return

        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag_left,
                z=0.1,
            )
        )

        if not self.plan_success:
            return

        tray_place_pose_left = self.tray.get_functional_point(0)

        self.move(
            self.place_actor(
                self.hamburg,
                arm_tag=arm_tag_left,
                target_pose=tray_place_pose_left,
                functional_point_id=0,
                constrain="free",
                pre_dis=0.1,
                pre_dis_axis="fp",
            )
        )

        if not self.plan_success:
            return

        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag_left,
                z=0.08,
            )
        )

        self.info["info"] = {
            "{A}": f"006_hamburg/base{self.object1_id}",
            "{B}": f"008_tray/base{self.tray_id}",
            "{C}": f"005_french-fries/base{self.object2_id}",
        }

    def _subtask2_motion_plan(self):
        """子任务2：右臂将薯条放到托盘上。"""
        arm_tag_right = ArmTag("right")

        self.move(
            self.grasp_actor(
                self.frenchfries,
                arm_tag=arm_tag_right,
                pre_grasp_dis=0.1,
            )
        )

        if not self.plan_success:
            return

        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag_right,
                z=0.1,
            )
        )

        if not self.plan_success:
            return

        tray_place_pose_right = self.tray.get_functional_point(1)

        self.move(
            self.place_actor(
                self.frenchfries,
                arm_tag=arm_tag_right,
                target_pose=tray_place_pose_right,
                functional_point_id=0,
                constrain="free",
                pre_dis=0.1,
                pre_dis_axis="fp",
            )
        )

        if not self.plan_success:
            return

        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag_right,
                z=0.08,
            )
        )

        self.info["info"] = {
            "{A}": f"006_hamburg/base{self.object1_id}",
            "{B}": f"008_tray/base{self.tray_id}",
            "{C}": f"005_french-fries/base{self.object2_id}",
        }

    def _check_subtask1_success(self):
        """子任务1成功条件：汉堡放到托盘 functional point 0，且左夹爪打开。"""
        dis1 = np.linalg.norm(
            self.tray.get_functional_point(0, "pose").p[0:2]
            - self.hamburg.get_functional_point(0, "pose").p[0:2]
        )

        threshold = 0.08
        return dis1 < threshold and self.is_left_gripper_open()

    def _check_subtask2_success(self):
        """子任务2成功条件：复用完整任务成功条件。"""
        return super().check_success()

    def check_success(self):
        """整体成功检查。"""
        return self.subtask_success[0] and self.subtask_success[1]

    def _reset_after_subtask(self, subtask_id=None):
        """Only reset between subtasks; never reset after the final subtask."""
        if subtask_id == 2:
            return True

        left_reset_success = self.move(self.back_to_origin(arm_tag="left"))
        right_reset_success = self.move(self.back_to_origin(arm_tag="right"))
        return bool(left_reset_success and right_reset_success and self.plan_success)
