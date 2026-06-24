from .place_dual_shoes import place_dual_shoes
import numpy as np


class place_dual_shoes_decomposition(place_dual_shoes):
    def __init__(self, **args):
        super().__init__(**args)
        self.current_subtask = 0
        self.subtask_done = [False, False]
        self.subtask_success = [False, False]

    def play_once(self):
        self._execute_subtask(1)
        if not self.subtask_success[0]:
            self.info["info"] = {
                "subtask1_success": False,
                "subtask2_success": False,
                "failed_subtask": 1,
                "failed_reason": "subtask1 failed: left shoe was not placed in the shoe box",
            }
            return self.info

        self._reset_after_subtask(subtask_id=1)

        self._execute_subtask(2)
        if not self.subtask_success[1]:
            print("[FAIL] Subtask 2 failed: right shoe was not placed in the shoe box")

        self.info["info"] = {
            "subtask1_success": self.subtask_success[0],
            "subtask2_success": self.subtask_success[1],
            "failed_subtask": None if self.check_success() else 2,
            "{A}": f"041_shoe/base{self.shoe_id}",
            "{B}": "007_shoe-box/base0",
        }
        return self.info

    def _execute_subtask(self, subtask_id):
        self.current_subtask = subtask_id
        self.subtask_done[subtask_id - 1] = False
        self.subtask_success[subtask_id - 1] = False

        self._setup_subtask_config(subtask_id)
        getattr(self, f"_subtask{subtask_id}_motion_plan")()

        self.subtask_success[subtask_id - 1] = getattr(
            self, f"_check_subtask{subtask_id}_success"
        )()
        self.subtask_done[subtask_id - 1] = True

    def _setup_subtask_config(self, subtask_id):
        self.current_subtask = subtask_id
        self.plan_success = True
        self.FRAME_IDX = 0
        self._clear_path_cache()

    def _clear_path_cache(self):
        self.left_joint_path = []
        self.right_joint_path = []
        self.left_cnt = 0
        self.right_cnt = 0

    def _subtask1_motion_plan(self):
        left_arm_tag = "left"

        self.move(
            self.grasp_actor(
                self.left_shoe,
                arm_tag=left_arm_tag,
                pre_grasp_dis=0.1,
            )
        )
        if not self.plan_success:
            return

        self.move(self.move_by_displacement(left_arm_tag, z=0.15))
        if not self.plan_success:
            return

        left_target = self.shoe_box.get_functional_point(0)
        left_place_pose = self.place_actor(
            self.left_shoe,
            target_pose=left_target,
            arm_tag=left_arm_tag,
            functional_point_id=0,
            pre_dis=0.07,
            dis=0.02,
            constrain="align",
        )
        self.move(left_place_pose)
        if not self.plan_success:
            return

        self.move(self.back_to_origin(left_arm_tag))

        self.delay(3)

        self.info["info"] = {
            "{A}": f"041_shoe/base{self.shoe_id}",
            "{B}": "007_shoe-box/base0",
        }

    def _subtask2_motion_plan(self):
        right_arm_tag = "right"

        self.move(
            self.grasp_actor(
                self.right_shoe,
                arm_tag=right_arm_tag,
                pre_grasp_dis=0.1,
            )
        )
        if not self.plan_success:
            return

        self.move(self.move_by_displacement(right_arm_tag, z=0.15))
        if not self.plan_success:
            return

        right_target = self.shoe_box.get_functional_point(1)
        right_place_pose = self.place_actor(
            self.right_shoe,
            target_pose=right_target,
            arm_tag=right_arm_tag,
            functional_point_id=0,
            pre_dis=0.07,
            dis=0.02,
            constrain="align",
        )
        self.move(right_place_pose)

        self.delay(3)

        self.info["info"] = {
            "{A}": f"041_shoe/base{self.shoe_id}",
            "{B}": "007_shoe-box/base0",
        }

    def _check_subtask1_success(self):
        left_shoe_pose_p = np.array(self.left_shoe.get_pose().p)
        left_shoe_pose_q = np.array(self.left_shoe.get_pose().q)
        if left_shoe_pose_q[0] < 0:
            left_shoe_pose_q *= -1

        target_pose_p = np.array([0, -0.13])
        target_pose_q = np.array([0.5, 0.5, -0.5, -0.5])
        eps = np.array([0.05, 0.05, 0.07, 0.07, 0.07, 0.07])

        return (
            np.all(abs(left_shoe_pose_p[:2] - (target_pose_p - [0, 0.04])) < eps[:2])
            and np.all(abs(left_shoe_pose_q - target_pose_q) < eps[-4:])
            and abs(left_shoe_pose_p[2] - (self.shoe_box.get_pose().p[2] + 0.01)) < 0.03
            and self.is_left_gripper_open()
        )

    def _check_subtask2_success(self):
        return super().check_success()

    def check_success(self):
        return self.subtask_success[0] and self.subtask_success[1]

    def _reset_after_subtask(self, subtask_id=None):
        if subtask_id == 2:
            return True

        left_reset_success = self.move(self.back_to_origin(arm_tag="left"))
        right_reset_success = self.move(self.back_to_origin(arm_tag="right"))
        return bool(left_reset_success and right_reset_success and self.plan_success)
