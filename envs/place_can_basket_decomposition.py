from envs.place_can_basket import place_can_basket
import numpy as np


class place_can_basket_decomposition(place_can_basket):
    def __init__(self, **args):
        super().__init__(**args)
        self.current_subtask = 0
        self.subtask_done = [False, False]
        self.subtask_success = [False, False]

    def setup_demo(self, is_test=False, **kwags):
        """覆盖父类方法，保存 kwags 参数供后续使用"""
        self.kwags = kwags

        super().setup_demo(is_test=is_test, **kwags)

    def play_once(self):
        """兼容常规执行：顺序执行两个独立的子任务"""
        self._execute_subtask(1)
        if not self.subtask_success[0]:
            print(f"[FAIL] 子任务1 失败（把罐子放进篮子）")
            self.info["info"] = {
                "subtask1_success": False,
                "subtask2_success": False,
                "failed_subtask": 1,
                "failed_reason": "子任务1失败：未能将罐子放入篮子",
            }
            return self.info

        self._execute_subtask(2)
        if not self.subtask_success[1]:
            print(f"[FAIL] 子任务2 失败（把篮子提起来）")

        self.info["info"] = {
            "subtask1_success": self.subtask_success[0],
            "subtask2_success": self.subtask_success[1],
            "failed_subtask": None if (self.subtask_success[0] and self.subtask_success[1]) else 2,
            "{A}": f"{self.can_name}/base{self.can_id}",
            "{B}": f"{self.basket_name}/base{self.basket_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def _execute_subtask(self, subtask_id):
        """环境内的独立执行逻辑（如果外部直接调 play_once）"""
        self.current_subtask = subtask_id
        self.subtask_done[subtask_id - 1] = False
        self.subtask_success[subtask_id - 1] = False

        self._setup_subtask_config(subtask_id)

        motion_plan_func = getattr(self, f"_subtask{subtask_id}_motion_plan")
        check_success_func = getattr(
            self, f"_check_subtask{subtask_id}_success")

        motion_plan_func()
        if subtask_id != len(self.subtask_success):
            self._reset_after_subtask(subtask_id=subtask_id)

        self.subtask_success[subtask_id - 1] = check_success_func()
        self.subtask_done[subtask_id - 1] = True

    def _setup_subtask_config(self, subtask_id):
        """每个子任务开始前：清理上一段的内存残余，切割视频时间轴"""
        self.current_subtask = subtask_id
        self.plan_success = True
        self.FRAME_IDX = 0
        self._clear_path_cache()
        if subtask_id == 1:
            self.subtask1_used_fallback_place = False

    def _clear_path_cache(self):
        """清空上一段动作的轨迹数组"""
        self.left_joint_path = []
        self.right_joint_path = []
        self.left_cnt = 0
        self.right_cnt = 0

    def _subtask1_motion_plan(self):
        """统一接口：子任务1动作规划"""
        return self._subtask1_place_can()

    def _subtask2_motion_plan(self):
        """统一接口：子任务2动作规划"""
        return self._subtask2_lift_basket()

    def _subtask1_place_can(self):
        """子任务1：抓取罐子并放入篮子"""
        self.move(self.grasp_actor(
            self.can, arm_tag=self.arm_tag, pre_grasp_dis=0.05
        ))

        if not self.plan_success:
            return

        place_pose = self.get_arm_pose(arm_tag=self.arm_tag)
        f0 = np.array(self.basket.get_functional_point(0))
        f1 = np.array(self.basket.get_functional_point(1))
        if np.linalg.norm(f0[:2] - place_pose[:2]) < np.linalg.norm(f1[:2] - place_pose[:2]):
            place_pose = f0.copy()
            place_pose[:2] = f0[:2]
            place_pose[3:] = ((-1, 0, 0, 0) if self.arm_tag ==
                              "left" else (0.05, 0, 0, 0.99))
        else:
            place_pose = f1.copy()
            place_pose[:2] = f1[:2]
            place_pose[3:] = ((-1, 0, 0, 0) if self.arm_tag ==
                              "left" else (0.05, 0, 0, 0.99))

        self.move(
            self.place_actor(
                self.can,
                arm_tag=self.arm_tag,
                target_pose=place_pose,
                dis=0.02,
                is_open=False,
                constrain="free",
            )
        )

        if not self.plan_success:
            self.plan_success = True
            self.subtask1_used_fallback_place = True
            place_pose[0] += -0.15 if self.arm_tag == "left" else 0.15
            place_pose[2] += 0.15
            self.move(self.move_to_pose(
                arm_tag=self.arm_tag, target_pose=place_pose))
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=-0.1))
            self.move(self.open_gripper(arm_tag=self.arm_tag))
            self.move(self.back_to_origin(arm_tag=self.arm_tag))
        else:
            self.move(self.open_gripper(arm_tag=self.arm_tag))
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))
            self.move(self.back_to_origin(arm_tag=self.arm_tag))

    def _subtask2_lift_basket(self):
        """子任务2：抬起篮子"""
        pre_grasp_dis = 0.02 if getattr(
            self, "subtask1_used_fallback_place", False
        ) else 0.08
        self.move(self.grasp_actor(
            self.basket,
            arm_tag=self.arm_tag.opposite,
            pre_grasp_dis=pre_grasp_dis,
        ))

        if not self.plan_success:
            return

        self.move(self.close_gripper(arm_tag=self.arm_tag.opposite))
        self.move(
            self.move_by_displacement(
                arm_tag=self.arm_tag.opposite,
                x=-0.02 if self.arm_tag.opposite == "left" else 0.02,
                z=0.05,
            )
        )

    def _check_subtask1_success(self):
        """子任务1成功条件"""
        can_p = self.can.get_pose().p
        basket_p = self.basket.get_pose().p
        can_contact_basket = self.check_actors_contact("071_can", "110_basket")
        return (
            np.sum(np.sqrt(np.power(can_p - basket_p, 2))) < 0.15 and
            can_contact_basket
        )

    def _check_subtask2_success(self):
        """子任务2成功条件"""
        return super().check_success()

    def check_success(self):
        """整体成功检查"""
        return self.subtask_success[0] and self.subtask_success[1]

    def _reset_after_subtask(self, subtask_id=None):
        """Only reset between subtasks; never reset after the final subtask."""
        if subtask_id == 2:
            return True

        left_reset_success = self.move(self.back_to_origin(arm_tag="left"))
        right_reset_success = self.move(self.back_to_origin(arm_tag="right"))
        return bool(left_reset_success and right_reset_success and self.plan_success)
