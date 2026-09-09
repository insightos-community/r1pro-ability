# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""原生 MuJoCo 产品链的显式 Ability 集成测试。

默认单元测试不会启动外部 Runtime；只有调用方传入运行中的 Scene Instance
与资产目录时才执行。本文件刻意直接调用现有七类 Ability 的公共 Service，
用于在接入 AbilityFramework/Pilot 之前先隔离 SDK、Provider 与真实接触问题。
"""

from __future__ import annotations

import os
import threading
import time
import unittest
from pathlib import Path

from r1pro_abilities import (
    AbilityRole,
    ExecutionStatus,
    ModelProfileRegistry,
    R1ProAbilityService,
)
from semantic_robot_sdk_core import BackendUnavailable
from semantic_robot_sdk_r1pro import create_mujoco_sdk


RUNTIME_URL = os.getenv("SEMANTIC_MUJOCO_RUNTIME_URL")
SCENE_INSTANCE_ID = os.getenv("SEMANTIC_MUJOCO_SCENE_INSTANCE_ID")
ROBOT_ID = os.getenv("SEMANTIC_MUJOCO_ROBOT_ID")
ASSET_ROOT = os.getenv("SEMANTIC_R1PRO_ASSET_ROOT")


@unittest.skipUnless(
    RUNTIME_URL and SCENE_INSTANCE_ID and ROBOT_ID and ASSET_ROOT,
    "需要显式提供运行中的原生 MuJoCo Scene Instance",
)
class MujocoProductAbilityTest(unittest.TestCase):
    def setUp(self) -> None:
        assert RUNTIME_URL and SCENE_INSTANCE_ID and ROBOT_ID and ASSET_ROOT
        asset_root = Path(ASSET_ROOT)
        self.sdk = create_mujoco_sdk(
            RUNTIME_URL,
            ROBOT_ID,
            asset_root
            / "robot/r1_pro_tote_gripper/meshes/r1_pro_tote_gripper.urdf",
            scene_instance_id=SCENE_INSTANCE_ID,
            # URDF 使用 package://r1_pro_chassis/...，因此 Pinocchio 需要包含
            # 该 package 目录的父目录，而不是整个 Asset 仓库根。
            package_directories=[str(asset_root / "robot")],
            timeout_seconds=15,
        )
        models = ModelProfileRegistry.load(
            Path(__file__).resolve().parents[2]
            / "semantic-robot-deployment/type-packages/r1pro-mujoco/templates/model-registry.json"
        )
        self.services = {
            role: R1ProAbilityService(role, self.sdk, models)
            for role in (
                AbilityRole.NAVIGATION,
                AbilityRole.MANIPULATOR_MOTION,
                AbilityRole.END_EFFECTOR,
                AbilityRole.OBJECT_PERCEPTION,
                AbilityRole.GRASP_PLANNING,
            )
        }

    def tearDown(self) -> None:
        for service in self.services.values():
            service.close()
        self.sdk.close()

    def _invoke(self, role: AbilityRole, task: str, data: dict) -> dict:
        result = self.services[role].invoke(task, data)
        if result["status"] not in {
            ExecutionStatus.ACCEPTED.value,
            ExecutionStatus.RUNNING.value,
        }:
            return result
        invocation_id = result["invocation_id"]
        # 双末端携物轨迹会在同步笛卡尔路点上重新求解并停稳，真实 Runtime
        # 的执行时间明显长于普通短 Action。这里只扩大物理 Gate 的观察窗口；
        # Runtime 与 Ability 仍使用轨迹自身的超时和终态，不会把卡死伪装成功。
        timeout_seconds = 180 if task == "LiftHeldObject" else 30
        deadline = time.monotonic() + timeout_seconds
        contact_trace: list[dict] = []
        sampling_sdk = self.sdk
        if task == "LiftHeldObject":
            # 诊断采样使用独立连接，确保采到的是物理过程而不是命令终态后的
            # 单个快照。采样频率刻意限制为 2Hz：SceneSnapshot 包含完整场景，
            # 高频读取会与 MuJoCo 控制循环争用实例锁，反而改变被测轨迹的时序。
            assert RUNTIME_URL and ROBOT_ID and ASSET_ROOT and SCENE_INSTANCE_ID
            asset_root = Path(ASSET_ROOT)
            sampling_sdk = create_mujoco_sdk(
                RUNTIME_URL,
                ROBOT_ID,
                asset_root
                / "robot/r1_pro_tote_gripper/meshes/r1_pro_tote_gripper.urdf",
                scene_instance_id=SCENE_INSTANCE_ID,
                package_directories=[str(asset_root / "robot")],
                timeout_seconds=15,
            )
        sampling_done = threading.Event()

        def sample_contacts() -> None:
            previous_signature: str | None = None
            while not sampling_done.wait(0.5 if task == "LiftHeldObject" else 0.02):
                try:
                    state = sampling_sdk.state.snapshot()
                except BackendUnavailable:
                    # 诊断采样不能改变被测执行的终态；Runtime 控制循环繁忙时
                    # 丢弃这一帧，主线程仍会按 command_id 完成正式状态对账。
                    continue
                sample = {
                    tool_ref: {
                        "position": round(tool.position, 6),
                        "hook_contact": tool.hook_contact,
                        "clamp_contact": tool.clamp_contact,
                        "hook_force_n": round(tool.hook_force_n, 3),
                        "clamp_force_n": round(tool.clamp_force_n, 3),
                        "hook_support_ratio": round(tool.hook_support_ratio, 3),
                        "hook_tangential_speed_m_s": round(
                            tool.hook_tangential_speed_m_s, 6
                        ),
                    }
                    for tool_ref, tool in state.tool_states.items()
                }
                if task == "LiftHeldObject":
                    try:
                        scene = sampling_sdk.state.scene_snapshot()
                    except BackendUnavailable:
                        continue
                    object_state = next(
                        (
                            item
                            for item in scene.objects
                            if item.source_id == str(data["object_ref"])
                        ),
                        None,
                    )
                    contact_trace.append(
                        {
                            "end_effector_z": {
                                key: round(value.position[2], 6)
                                for key, value in state.end_effectors.items()
                            },
                            "object_z": round(object_state.pose.position[2], 6)
                            if object_state
                            else None,
                            "tools": sample,
                        }
                    )
                    continue
                signature = repr(
                    {
                        tool_ref: (
                            value["hook_contact"],
                            value["clamp_contact"],
                            value["hook_force_n"] > 0,
                            value["hook_support_ratio"] >= 0.6,
                        )
                        for tool_ref, value in sample.items()
                    }
                )
                if signature != previous_signature:
                    contact_trace.append({"signature": signature, "tools": sample})
                    previous_signature = signature

        sampler = None
        if task in {"CloseUntilContact", "LiftHeldObject"}:
            # Runtime 的 command feedback 是阻塞式长轮询；另一个短线程只在
            # 测试中采集接触状态变化，避免把诊断逻辑写进生产 SDK。
            sampler = threading.Thread(target=sample_contacts, daemon=True)
            sampler.start()
        while time.monotonic() < deadline:
            result = self.services[role].invoke(
                "GetExecution", {"invocation_id": invocation_id}
            )
            if result["status"] not in {
                ExecutionStatus.ACCEPTED.value,
                ExecutionStatus.RUNNING.value,
            }:
                sampling_done.set()
                if sampler is not None:
                    sampler.join(timeout=1)
                if sampling_sdk is not self.sdk:
                    sampling_sdk.close()
                if contact_trace:
                    result["contact_trace"] = contact_trace
                return result
            # Ability GetExecution 会读取一次 SDK feedback 和一次 command 状态。
            # 真实物理 Gate 只需 2Hz 观测，避免测试轮询本身压低仿真步进速度。
            time.sleep(0.5)
        sampling_done.set()
        if sampler is not None:
            sampler.join(timeout=1)
        if sampling_sdk is not self.sdk:
            sampling_sdk.close()
        self.fail(f"{task} 在 {timeout_seconds} 秒内没有终态: {result}")

    @staticmethod
    def _pose(value, revision: str) -> dict:
        return {
            "frame_id": value.frame_id,
            "position_m": list(value.position),
            "orientation_xyzw": list(value.quaternion_xyzw),
            "observed_at": None,
            "revision": revision,
        }

    def test_direct_bilateral_reaches_real_hook_contact(self) -> None:
        """产品候选必须靠真实接触结束插入，不能用关节到位冒充预抓取。"""

        snapshot = self.sdk.state.scene_snapshot()
        source = next(item for item in snapshot.objects if item.source_id == "tote-large-smoke")
        approach = next(
            item for item in snapshot.regions if item.source_id == "pallet-a-approach"
        )
        revision = f"scene-{snapshot.generation}-{snapshot.observed_at.isoformat()}"
        target = {
            "target_ref": approach.source_id,
            "pose": self._pose(approach.pose, revision),
            "constraints": {},
        }
        planned = self._invoke(
            AbilityRole.NAVIGATION,
            "PlanRoute",
            {
                "invocation_id": "product-plan-source",
                "robot_id": ROBOT_ID,
                "target": target,
                "navigation_purpose": "approach_grasp",
                "maximum_speed_mps": 0.5,
                "minimum_clearance_m": 0.2,
                "carrying_object": None,
            },
        )
        self.assertEqual(planned["status"], ExecutionStatus.SUCCEEDED.value, planned)
        followed = self._invoke(
            AbilityRole.NAVIGATION,
            "FollowRoute",
            {
                "invocation_id": "product-follow-source",
                "robot_id": ROBOT_ID,
                "route_ref": planned["result"]["route_ref"],
                "navigation_purpose": "approach_grasp",
                "maximum_speed_mps": 0.5,
                "minimum_clearance_m": 0.2,
                "carrying_object": None,
            },
        )
        self.assertEqual(followed["status"], ExecutionStatus.SUCCEEDED.value, followed)

        located = self._invoke(
            AbilityRole.OBJECT_PERCEPTION,
            "LocateObject",
            {
                "invocation_id": "product-locate-source",
                "robot_id": ROBOT_ID,
                "object_ref": source.source_id,
                "minimum_confidence": 0.9,
                "model_profile": "mujoco-ground-truth",
            },
        )
        self.assertEqual(located["status"], ExecutionStatus.SUCCEEDED.value, located)
        observed = located["result"]["verification"]
        candidates = self._invoke(
            AbilityRole.GRASP_PLANNING,
            "GenerateCandidates",
            {
                "invocation_id": "product-generate-candidates",
                "robot_id": ROBOT_ID,
                "object_ref": source.source_id,
                "target_pose": observed["pose"],
                "object_extent_m": observed["extent_m"],
                "target_revision": located["observations"][0]["revision"],
                "preferred_strategy": "direct_bilateral",
                "maximum_candidates": 3,
                "model_profile": "mujoco-tote-grasp",
            },
        )
        self.assertEqual(candidates["status"], ExecutionStatus.SUCCEEDED.value, candidates)
        candidate = next(
            item
            for item in candidates["result"]["candidates"]
            if item["strategy"] == "direct_bilateral"
        )
        opened = self._invoke(
            AbilityRole.END_EFFECTOR,
            "SetOpening",
            {
                "invocation_id": "product-open-tools",
                "robot_id": ROBOT_ID,
                "tools": candidate["opening_setpoints"],
            },
        )
        self.assertEqual(opened["status"], ExecutionStatus.SUCCEEDED.value, opened)
        self.assertTrue(opened["result"]["verified"], opened)

        for index, (targets, purpose, speed) in enumerate(
            (
                (candidate["transfer_poses"], "pregrasp", 0.15),
                (candidate["approach_poses"], "pregrasp", 0.15),
                (candidate["hook_insert_poses"], "insert", 0.08),
                (candidate["hook_seat_poses"], "seat", 0.04),
            ),
            start=1,
        ):
            moved = self._invoke(
                AbilityRole.MANIPULATOR_MOTION,
                "MoveEndEffector",
                {
                    "invocation_id": f"product-direct-move-{index}",
                    "robot_id": ROBOT_ID,
                    "targets": targets,
                    "coordination": "synchronized",
                    "purpose": purpose,
                    "object_ref": source.source_id,
                    "candidate_id": candidate["candidate_id"],
                    "target_revision": candidate["observation_revision"],
                    "maximum_speed_mps": speed,
                    "max_contact_force_n": (
                        60.0 if purpose in {"insert", "seat"} else None
                    ),
                },
            )
            self.assertEqual(moved["status"], ExecutionStatus.SUCCEEDED.value, moved)

        verified = self._invoke(
            AbilityRole.OBJECT_PERCEPTION,
            "VerifyPregrasp",
            {
                "invocation_id": "product-verify-pregrasp",
                "robot_id": ROBOT_ID,
                "object_ref": source.source_id,
                "candidate_id": candidate["candidate_id"],
                "planned_object_pose": candidate["planned_object_pose"],
                "expected_targets": candidate["hook_seat_poses"],
                "target_revision": candidate["observation_revision"],
                "maximum_position_error_m": 0.05,
                "model_profile": "mujoco-ground-truth",
            },
        )
        self.assertEqual(verified["status"], ExecutionStatus.SUCCEEDED.value, verified)
        self.assertTrue(verified["result"]["verification"]["reached"], verified)

        # 直接双侧策略必须在同一个 Ability Execution 中同时下发两侧命令。
        # Runtime 仍分别执行两个 gripper command，并不提供“原子双夹具”能力；
        # 但由 Ability 聚合等待两侧终态，可避免先闭合的一侧单独推动自由箱体。
        close_started = self.services[AbilityRole.END_EFFECTOR].invoke(
            "CloseUntilContact",
            {
                "invocation_id": "product-close-bilateral",
                "robot_id": ROBOT_ID,
                "object_ref": source.source_id,
                "tools": candidate["clamp_setpoints"],
                "candidate_id": candidate["candidate_id"],
                "grasp_pose": observed["pose"],
            },
        )
        self.assertIn(
            close_started["status"],
            {ExecutionStatus.ACCEPTED.value, ExecutionStatus.RUNNING.value, ExecutionStatus.SUCCEEDED.value},
        )
        contact_trace = []
        previous_signature = None
        sample_deadline = time.monotonic() + 5
        while time.monotonic() < sample_deadline:
            state = self.sdk.state.snapshot()
            sample = {
                tool_ref: {
                    "position": round(tool.position, 6),
                    "hook_contact": tool.hook_contact,
                    "clamp_contact": tool.clamp_contact,
                    "hook_force_n": round(tool.hook_force_n, 3),
                    "clamp_force_n": round(tool.clamp_force_n, 3),
                    "hook_support_ratio": round(tool.hook_support_ratio, 3),
                    "hook_tangential_speed_m_s": round(
                        tool.hook_tangential_speed_m_s, 6
                    ),
                }
                for tool_ref, tool in state.tool_states.items()
            }
            signature = repr(
                {
                    ref: (
                        value["hook_contact"],
                        value["clamp_contact"],
                        value["hook_force_n"] > 0,
                        value["hook_support_ratio"] >= 0.6,
                    )
                    for ref, value in sample.items()
                }
            )
            if signature != previous_signature:
                contact_trace.append(sample)
                previous_signature = signature
            time.sleep(0.02)
        final_contact = self.services[AbilityRole.END_EFFECTOR].invoke(
            "GetExecution",
            {"invocation_id": "product-close-bilateral"},
        )
        final_contact["contact_trace"] = contact_trace
        self.assertEqual(
            final_contact["status"], ExecutionStatus.SUCCEEDED.value, final_contact
        )
        self.assertTrue(
            final_contact["observations"][0]["value"]["stable_bilateral_load"],
            final_contact,
        )

        lifted = self._invoke(
            AbilityRole.MANIPULATOR_MOTION,
            "LiftHeldObject",
            {
                "invocation_id": "product-lift-held-object",
                "robot_id": ROBOT_ID,
                "object_ref": source.source_id,
                "tools": [item["tool_ref"] for item in candidate["tool_targets"]],
                "candidate_id": candidate["candidate_id"],
                "distance_m": 0.08,
                # 双侧夹具共同承载周转箱时，末端轨迹必须给两侧关节控制器
                # 足够的同步跟随时间。这里约束的是携物抬升速度，而不是放宽
                # Runtime 的关节到位精度；速度过高会让两侧瞬时位姿差把箱体
                # 从侧面凹槽中扭出，真实物理链应明确失败并进入 hold。
                "maximum_speed_mps": 0.04,
            },
        )
        self.assertEqual(
            lifted["status"],
            ExecutionStatus.SUCCEEDED.value,
            {
                "status": lifted["status"],
                "error": lifted.get("error"),
                # 真实物理失败时，完整 Feedback 往往包含数百个轮询点。断言只
                # 保留接触轨迹首尾，既能看出箱体何时脱钩，也避免 Gate 日志被
                # 重复 running 反馈淹没。完整序列仍保存在 Ability Execution。
                "contact_trace_head": lifted.get("contact_trace", [])[:3],
                "contact_trace_tail": lifted.get("contact_trace", [])[-8:],
            },
        )

        grasp = self._invoke(
            AbilityRole.OBJECT_PERCEPTION,
            "VerifyGrasp",
            {
                "invocation_id": "product-verify-grasp",
                "robot_id": ROBOT_ID,
                "object_ref": source.source_id,
                "tools": [item["tool_ref"] for item in candidate["tool_targets"]],
                "candidate_id": candidate["candidate_id"],
                "initial_object_pose": observed["pose"],
                "minimum_lift_height_m": 0.08,
                "stable_duration_ms": 500,
                "model_profile": "mujoco-ground-truth",
            },
        )
        self.assertEqual(grasp["status"], ExecutionStatus.SUCCEEDED.value, grasp)
        self.assertTrue(grasp["result"]["verification"]["held"], grasp)


if __name__ == "__main__":
    unittest.main()
