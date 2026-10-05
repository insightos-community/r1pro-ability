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

import math
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from pydantic import ValidationError
from r1pro_abilities import (
    AbilityRole,
    ExecutionStatus,
    ModelProfileError,
    ModelProfileRegistry,
    R1ProAbilityService,
)
from r1pro_abilities.ability_utils import scene_collision_environment
from r1pro_abilities.handlers.grasp_planning import (
    _candidate,
    _candidate_reachability,
    _order_strategies_for_scene,
    _tote_collision_parts,
)
from r1pro_abilities.handlers.manipulator_motion import ManipulatorMotionHandler
from r1pro_abilities.handlers.object_perception import (
    _groove_engagement_error,
    _lift_height_meets_requirement,
    _placement_lateral_clearances,
    _position_in_object_frame,
    _resolve,
)
from r1pro_abilities.tool_load import (
    DEFAULT_TOOL_LOAD_POLICY,
    ToolLoadAssessment,
    ToolLoadPolicy,
    wait_for_stable_tool_load,
)
from semantic_robot_sdk_core import (
    BackendUnavailable,
    CommandState,
    EnvironmentCollisionObject,
    PlanningError,
    Pose,
    RobotDeployment,
    SceneObject,
    SceneRegion,
    SceneSnapshot,
    SensorFrame,
)
from semantic_robot_sdk_r1pro import R1ProSDK


ROOT = Path(__file__).resolve().parents[1]
POSE = {
    "frame_id": "world",
    "position_m": [1.0, 2.0, 0.0],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "observed_at": "2026-08-09T00:00:00+00:00",
    "revision": "pose-r1",
}
TARGET = {
    "target_ref": "map://target/pallet-b",
    "pose": POSE,
    "constraints": {},
}


class AbilityServiceTest(unittest.TestCase):
    def test_pregrasp_relative_geometry_detects_tote_pushed_with_tool(self) -> None:
        """末端到位但箱体被一起推走时，不能冒充下钩已经进入凹槽。"""

        planned_relative = _position_in_object_frame(
            (-0.293, 1.29, 1.063),
            (0.0, 1.29, 0.999),
            (0.0, 0.0, 0.0, 1.0),
        )
        actual_relative = _position_in_object_frame(
            (-0.293, 1.29, 1.063),
            (-0.020, 1.29, 0.999),
            (0.0, 0.0, 0.0, 1.0),
        )

        self.assertAlmostEqual(
            math.dist(planned_relative, actual_relative), 0.020, places=6
        )

    def test_pregrasp_allows_motion_along_same_side_groove(self) -> None:
        """槽长方向漂移不应阻止后续seat，深度或高度偏差仍会被拒绝。"""

        expected = (0.293, 0.0, 0.064)
        self.assertAlmostEqual(
            _groove_engagement_error((0.293, -0.012, 0.064), expected),
            0.0,
            places=9,
        )
        self.assertAlmostEqual(
            _groove_engagement_error((0.298, 0.0, 0.064), expected),
            0.005,
            places=9,
        )
        self.assertAlmostEqual(
            _groove_engagement_error((0.293, 0.0, 0.058), expected),
            0.006,
            places=9,
        )

    def test_tote_collision_proxy_preserves_open_grooves(self) -> None:
        """目标周转箱不能被实心AABB堵住真实凹槽和箱内自由空间。"""

        target = EnvironmentCollisionObject(
            source_id="tote-large-l3-r1-c1",
            shape="box",
            pose=Pose(
                frame_id="world",
                position=(0.0, 0.0, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            size_xyz=(0.6, 0.4, 0.34),
            allowed_contact_end_effectors=("left",),
        )
        parts = _tote_collision_parts(
            target,
            {
                "extent_m": (0.6, 0.4, 0.34),
                "groove_axis_local": (1.0, 0.0, 0.0),
                "groove_width_m": 0.230,
                "groove_height_m": 0.030,
                "groove_depth_m": 0.026,
                "groove_top_offset_m": 0.086,
                "wall_thickness_m": 0.004,
            },
        )

        self.assertNotIn(target.source_id, {item.source_id for item in parts})
        by_id = {item.source_id: item for item in parts}
        left_ceiling = by_id["tote-large-l3-r1-c1#collision/groove-ceiling--1"]
        right_ceiling = by_id["tote-large-l3-r1-c1#collision/groove-ceiling-+1"]
        self.assertEqual(left_ceiling.size_xyz, (0.026, 0.230, 0.004))
        self.assertEqual(right_ceiling.size_xyz, (0.026, 0.230, 0.004))
        self.assertAlmostEqual(left_ceiling.pose.position[0], -0.287)
        self.assertAlmostEqual(right_ceiling.pose.position[0], 0.287)
        self.assertAlmostEqual(left_ceiling.pose.position[2], 0.086)
        self.assertEqual(
            left_ceiling.allowed_contact_end_effectors,
            ("left",),
        )
        # 分解结果只表达真实底面、箱壁和凹槽边界，不再存在覆盖箱内自由
        # 空间的0.6×0.4×0.34实心碰撞块。
        self.assertFalse(
            any(item.size_xyz == target.size_xyz for item in parts),
            parts,
        )

    def test_scene_collision_environment_uses_runtime_geometry_and_exclusion(
        self,
    ) -> None:
        """运动碰撞使用当前SceneSnapshot，并只排除明确接触的目标物体。"""

        sdk = self._sdk()
        observed_at = datetime.fromisoformat("2026-08-20T00:00:00+00:00")
        objects = [
            SceneObject(
                source_id=source_id,
                category="tote",
                name=source_id,
                pose=Pose(
                    frame_id="world",
                    position=(float(index), 1.0, 0.3),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                extent=(0.6, 0.4, 0.34),
            )
            for index, source_id in enumerate(("target", "neighbor"))
        ]
        snapshot = SceneSnapshot(
            scene_key="depalletizing",
            instance_id="scene-1",
            generation=1,
            coordinate_frame="world",
            objects=objects,
            regions=[],
            observed_at=observed_at,
        )
        try:
            with patch.object(sdk.state, "scene_snapshot", return_value=snapshot):
                environment = scene_collision_environment(
                    sdk, exclude_source_ids=("target",)
                )
            self.assertIsNotNone(environment)
            assert environment is not None
            self.assertEqual(environment.frame_id, "world")
            self.assertEqual(
                [item.source_id for item in environment.objects], ["neighbor"]
            )
            self.assertEqual(environment.objects[0].size_xyz, (0.6, 0.4, 0.34))
        finally:
            sdk.close()

    def test_lift_height_uses_observation_tolerance_not_simulation_steps(self) -> None:
        self.assertTrue(_lift_height_meets_requirement(0.079911, 0.08, 0.001))
        self.assertFalse(_lift_height_meets_requirement(0.075, 0.08, 0.001))

    def test_close_waits_for_real_contact_settling_before_success(self) -> None:
        """闭合后的相对运动先沉降，不能被单次采样直接判成抓取失败。"""

        sdk = self._sdk(defer_commands=False)
        state = sdk.state.snapshot()
        moving = ToolLoadAssessment(
            engaged=True,
            slipping=True,
            overloaded=False,
            sensor_fault=False,
            reasons=("component://tool/right:relative_motion_high",),
            tool_states=dict(state.tool_states),
        )
        stable = ToolLoadAssessment(
            engaged=True,
            slipping=False,
            overloaded=False,
            sensor_fault=False,
            reasons=(),
            tool_states=dict(state.tool_states),
        )
        try:
            with (
                patch(
                    "r1pro_abilities.tool_load.read_tool_load",
                    side_effect=[
                        (state, moving),
                        (state, moving),
                        (state, stable),
                        (state, stable),
                    ],
                ),
                patch(
                    "r1pro_abilities.tool_load.monotonic",
                    side_effect=[0.0, 0.0, 0.1, 0.2, 0.34],
                ),
                patch("r1pro_abilities.tool_load.sleep"),
            ):
                _, assessment, observed_ms = wait_for_stable_tool_load(
                    sdk,
                    ("component://tool/left", "component://tool/right"),
                    ToolLoadPolicy(stable_duration_s=0.12, acquisition_timeout_s=1.0),
                )
            self.assertTrue(assessment.safe)
            self.assertEqual(observed_ms, 140)
        finally:
            sdk.close()

    def _sdk(
        self, *, stop_confirmed: bool = True, defer_commands: bool = True
    ) -> R1ProSDK:
        value = yaml.safe_load(
            (ROOT / "configs/robot-deployment.fake.yaml").read_text(encoding="utf-8")
        )
        value["robot"]["sdk"]["options"].update(
            {
                "auto_complete": not defer_commands,
                "stop_confirmed": stop_confirmed,
            }
        )
        return R1ProSDK.from_deployment(RobotDeployment.model_validate(value))

    def _engage_bilateral_tools(
        self, sdk: R1ProSDK, object_ref: str = "object://box-1"
    ) -> None:
        """只通过公开夹具命令构造双侧原始接触，不注入业务持物结论。"""

        for side in ("left", "right"):
            sdk.backend.configure_grasp_fixture(
                side,
                object_ref,
                contact_opening_m=0.021,
                contact_force_n=7.0,
                minimum_holding_force_n=2.0,
            )
            command_id = f"test-engage-{side}"
            command = sdk.end_effector.close_until_contact(
                side,
                command_id=command_id,
                force_limit_n=5.0,
            )
            if command.status.value in {"accepted", "running"}:
                sdk.backend.complete_command(command_id)

    @staticmethod
    def _route_request(route_ref: str, invocation_id: str) -> dict:
        return {
            "invocation_id": invocation_id,
            "robot_id": "r1pro-fake-001",
            "route_ref": route_ref,
            "navigation_purpose": "transit",
            "maximum_speed_mps": 0.5,
            "minimum_clearance_m": 0.2,
        }

    def _plan(
        self, service: R1ProAbilityService, invocation_id: str | None = None
    ) -> str:
        planned = service.invoke(
            "PlanRoute",
            {
                "robot_id": "r1pro-fake-001",
                "invocation_id": invocation_id,
                "target": TARGET,
                "navigation_purpose": "transit",
                "maximum_speed_mps": 0.5,
                "minimum_clearance_m": 0.2,
            },
        )
        self.assertEqual(planned["status"], ExecutionStatus.SUCCEEDED.value)
        self.assertNotIn("_route", planned["result"])
        return planned["result"]["route_ref"]

    def test_route_identity_changes_with_runtime_generation(self) -> None:
        sdk = self._sdk()
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        try:
            first_ref = self._plan(service, "plan-generation-1")
            snapshot = sdk.backend.export_snapshot()
            snapshot.state.generation += 1
            sdk.backend.restore_snapshot(snapshot)

            second_ref = self._plan(service, "plan-generation-2")

            self.assertNotEqual(first_ref, second_ref)
            self.assertTrue(first_ref.startswith("route-g1-"))
            self.assertTrue(second_ref.startswith("route-g2-"))
            started = service.invoke(
                "FollowRoute", self._route_request(second_ref, "run-generation-2")
            )
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)
        finally:
            service.close()
            sdk.close()

    @staticmethod
    def _grasp_planning_request() -> dict:
        return {
            "robot_id": "r1pro-fake-001",
            "object_ref": "object://box-1",
            "target_pose": POSE,
            "object_extent_m": [0.6, 0.4, 0.34],
            "target_revision": "target-r1",
            "preferred_strategy": "auto",
            "maximum_candidates": 3,
            "model_profile": "mujoco-tote-grasp",
        }

    def test_physical_task_reports_feedback_and_real_stop(self) -> None:
        sdk = self._sdk()
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        try:
            route_ref = self._plan(service)
            started = service.invoke(
                "FollowRoute", self._route_request(route_ref, "run-1")
            )
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)
            self.assertEqual([item["sequence"] for item in started["feedback"]], [1, 2])

            stopped = service.invoke(
                "StopExecution", {"invocation_id": "run-1", "reason": "user stop"}
            )

            self.assertEqual(stopped["status"], ExecutionStatus.STOPPED.value)
            self.assertTrue(stopped["result"]["stop_evidence"]["holding"])
            self.assertTrue(sdk.state.snapshot().in_hold)
        finally:
            service.close()
            sdk.close()

    def test_async_result_is_only_published_after_command_success(self) -> None:
        sdk = self._sdk()
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        try:
            route_ref = self._plan(service)
            started = service.invoke(
                "FollowRoute", self._route_request(route_ref, "run-failed")
            )
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)
            self.assertEqual(started["result"], {})

            sdk.backend.interrupt_command(
                started["command_id"], "runtime command failed"
            )
            failed = service.invoke("GetExecution", {"invocation_id": "run-failed"})
            self.assertEqual(failed["status"], ExecutionStatus.INTERRUPTED.value)
            self.assertEqual(failed["result"], {})
            self.assertEqual(failed["error"]["code"], "ROBOT_COMMAND_INTERRUPTED")
        finally:
            service.close()
            sdk.close()

    def test_completed_short_command_keeps_ordered_feedback(self) -> None:
        sdk = self._sdk(defer_commands=False)
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        try:
            route_ref = self._plan(service)
            completed = service.invoke(
                "FollowRoute", self._route_request(route_ref, "run-completed")
            )
            self.assertEqual(completed["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertEqual(
                [item["sequence"] for item in completed["feedback"]], [1, 2]
            )
        finally:
            service.close()
            sdk.close()

    def test_unconfirmed_stop_is_interrupted(self) -> None:
        sdk = self._sdk(stop_confirmed=False)
        service = R1ProAbilityService(AbilityRole.END_EFFECTOR, sdk)
        try:
            service.invoke(
                "CloseUntilContact",
                {
                    "invocation_id": "grasp-1",
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "tools": [
                        {
                            "tool_ref": "component://tool/left",
                            "target_position_m": 0.01,
                            "maximum_force_n": 10.0,
                            "hold": True,
                        }
                    ],
                    "candidate_id": "candidate-1",
                    "grasp_pose": POSE,
                },
            )
            stopped = service.invoke("StopExecution", {"invocation_id": "grasp-1"})
            self.assertEqual(stopped["status"], ExecutionStatus.INTERRUPTED.value)
            self.assertEqual(stopped["error"]["code"], "STOP_UNCONFIRMED")
        finally:
            service.close()
            sdk.close()

    def test_close_only_confirms_safe_tool_motion_before_seating(self) -> None:
        """闭合不冒充承载；seat后的真实时间承载验证由RobotState负责。"""
        sdk = self._sdk(defer_commands=False)
        service = R1ProAbilityService(AbilityRole.END_EFFECTOR, sdk)
        tools = [
            {
                "tool_ref": f"component://tool/{side}",
                "target_position_m": 0.0,
                "maximum_force_n": 18.0,
                "hold": True,
            }
            for side in ("left", "right")
        ]
        try:
            for side in ("left", "right"):
                sdk.backend.configure_grasp_fixture(
                    side,
                    "object://box-1",
                    contact_opening_m=0.06,
                    contact_force_n=18.0,
                )
            result = service.invoke(
                "CloseUntilContact",
                {
                    "invocation_id": "stable-contact",
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "tools": tools,
                    "candidate_id": "candidate-1",
                    "grasp_pose": POSE,
                },
            )
            self.assertEqual(result["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertTrue(result["result"]["verified"])
            self.assertEqual(
                result["observations"][-1]["kind"], "manipulation.tool_closed"
            )
            value = result["observations"][-1]["value"]
            self.assertTrue(value["closure_completed"])
            self.assertNotIn("stable_contact_steps", value)
            self.assertNotIn("stable_bilateral_load", value)
        finally:
            service.close()
            sdk.close()

    def test_close_does_not_compare_hook_load_with_clamp_command_limit(self) -> None:
        """承重钩支撑力不是压紧执行器力，过载留给VerifyToolLoad判断。"""

        sdk = self._sdk(defer_commands=False)
        service = R1ProAbilityService(AbilityRole.END_EFFECTOR, sdk)
        tool_ref = "component://tool/left"
        try:
            sdk.backend.configure_grasp_fixture(
                "left",
                "object://box-1",
                contact_opening_m=0.06,
                contact_force_n=18.0,
            )
            observed = sdk.state.snapshot()
            left = observed.tool_states[tool_ref].model_copy(
                update={
                    "clamp_contact": True,
                    "clamp_force_n": 10.0,
                    "hook_contact": True,
                    "hook_force_n": 30.0,
                    "reached_target": False,
                    "sensor_fault": False,
                }
            )
            with patch.object(
                sdk.state,
                "snapshot",
                return_value=observed.model_copy(
                    update={"tool_states": {**observed.tool_states, tool_ref: left}}
                ),
            ):
                result = service.invoke(
                    "CloseUntilContact",
                    {
                        "invocation_id": "hook-load-above-clamp-limit",
                        "robot_id": "r1pro-fake-001",
                        "object_ref": "object://box-1",
                        "tools": [
                            {
                                "tool_ref": tool_ref,
                                "target_position_m": 0.0,
                                "maximum_force_n": 18.0,
                                "hold": True,
                            }
                        ],
                        "candidate_id": "candidate-1",
                        "grasp_pose": POSE,
                    },
                )
            self.assertEqual(result["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertTrue(result["result"]["verified"])
        finally:
            service.close()
            sdk.close()

    def test_close_accepts_contact_when_measured_force_overshoots_command(self) -> None:
        """接触后的瞬时测量可略高于命令限值，工具额定过载由VerifyToolLoad判断。"""

        sdk = self._sdk(defer_commands=False)
        service = R1ProAbilityService(AbilityRole.END_EFFECTOR, sdk)
        tool_ref = "component://tool/left"
        try:
            sdk.backend.configure_grasp_fixture(
                "left",
                "object://box-1",
                contact_opening_m=0.06,
                contact_force_n=18.0,
            )
            observed = sdk.state.snapshot()
            left = observed.tool_states[tool_ref].model_copy(
                update={
                    "clamp_contact": True,
                    "clamp_force_n": 60.34,
                    "hook_contact": True,
                    "hook_force_n": 70.18,
                    "reached_target": False,
                    "sensor_fault": False,
                }
            )
            with patch.object(
                sdk.state,
                "snapshot",
                return_value=observed.model_copy(
                    update={"tool_states": {**observed.tool_states, tool_ref: left}}
                ),
            ):
                result = service.invoke(
                    "CloseUntilContact",
                    {
                        "invocation_id": "contact-force-overshoot",
                        "robot_id": "r1pro-fake-001",
                        "object_ref": "object://box-1",
                        "tools": [
                            {
                                "tool_ref": tool_ref,
                                "target_position_m": 0.0,
                                "maximum_force_n": 60.0,
                                "hold": True,
                            }
                        ],
                        "candidate_id": "candidate-1",
                        "grasp_pose": POSE,
                    },
                )
            self.assertEqual(result["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertTrue(result["result"]["verified"])
        finally:
            service.close()
            sdk.close()

    def test_hold_remains_hold_when_execution_is_read_again(self) -> None:
        """同步 Hold 的后续 GetExecution 不得被 Release 收敛逻辑覆盖。"""
        sdk = self._sdk(defer_commands=False)
        service = R1ProAbilityService(AbilityRole.END_EFFECTOR, sdk)
        try:
            held = service.invoke(
                "HoldObject",
                {
                    "invocation_id": "hold-object",
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "tools": [],
                    "reason": "test",
                },
            )
            refreshed = service.invoke("GetExecution", {"invocation_id": "hold-object"})
            self.assertEqual(held["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertEqual(refreshed["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertIsNone(refreshed["error"])
            self.assertEqual(
                refreshed["observations"][0]["kind"], "manipulation.tool_hold"
            )
            self.assertTrue(refreshed["result"]["safe"])
        finally:
            service.close()
            sdk.close()

    def test_sensor_capture_writes_real_frame_for_pilot_publication(self) -> None:
        sdk = self._sdk(defer_commands=False)
        generation = sdk.state.snapshot().generation
        sdk.backend.publish_sensor_frame(
            SensorFrame(
                sensor_id="rgb",
                sequence=7,
                generation=generation,
                frame_id="camera_rgb_optical_frame",
                encoding="jpeg",
                payload=b"real-jpeg-frame",
                width=640,
                height=480,
            )
        )
        with tempfile.TemporaryDirectory() as artifact_root:
            service = R1ProAbilityService(
                AbilityRole.SENSOR_CAPTURE,
                sdk,
                artifact_exchange_root=artifact_root,
            )
            try:
                completed = service.invoke(
                    "CaptureRGBD",
                    {
                        "robot_id": "r1pro-fake-001",
                        "sensor_ids": ["rgb"],
                    },
                )
                self.assertEqual(completed["status"], ExecutionStatus.SUCCEEDED.value)
                candidate = completed["result"]["artifact_candidates"][0]
                self.assertEqual(candidate["publication_state"], "ready_for_pilot")
                self.assertFalse(Path(candidate["exchange_path"]).is_absolute())
                captured = Path(artifact_root, candidate["exchange_path"])
                self.assertEqual(captured.read_bytes(), b"real-jpeg-frame")
                self.assertEqual(completed["result"]["artifact_refs"], [])
            finally:
                service.close()
        sdk.close()

    def test_duplicate_invocation_requires_identical_content(self) -> None:
        sdk = self._sdk()
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        try:
            route_ref = self._plan(service)
            request = self._route_request(route_ref, "run-duplicate")
            first = service.invoke("FollowRoute", request)
            second = service.invoke("FollowRoute", request)
            self.assertEqual(first["command_id"], second["command_id"])
            with self.assertRaises(ValueError):
                service.invoke("FollowRoute", {**request, "maximum_speed_mps": 0.8})
        finally:
            service.close()
            sdk.close()

    def test_execution_store_does_not_replay_unknown_command_after_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "executions.sqlite"
            first_sdk = self._sdk()
            first = R1ProAbilityService(
                AbilityRole.NAVIGATION, first_sdk, execution_store_path=database
            )
            route_ref = self._plan(first)
            first.invoke("FollowRoute", self._route_request(route_ref, "restore-1"))
            first.close()
            first_sdk.close()

            second_sdk = self._sdk()
            second = R1ProAbilityService(
                AbilityRole.NAVIGATION, second_sdk, execution_store_path=database
            )
            try:
                restored = second.invoke("GetExecution", {"invocation_id": "restore-1"})
                self.assertEqual(restored["status"], ExecutionStatus.INTERRUPTED.value)
                self.assertEqual(restored["error"]["code"], "COMMAND_UNKNOWN")
            finally:
                second.close()
                second_sdk.close()

    def test_transient_runtime_read_failure_keeps_execution_reconcilable(self) -> None:
        sdk = self._sdk()
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        try:
            route_ref = self._plan(service)
            request = self._route_request(route_ref, "transient-runtime-read")
            started = service.invoke("FollowRoute", request)
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)

            with patch.object(
                sdk.commands,
                "feedback",
                side_effect=BackendUnavailable("一次状态读取超时"),
            ):
                unavailable = service.invoke(
                    "GetExecution", {"invocation_id": "transient-runtime-read"}
                )
            self.assertEqual(unavailable["status"], ExecutionStatus.RUNNING.value)
            self.assertEqual(
                unavailable["phase"], "command_status_temporarily_unavailable"
            )

            reconciled = service.invoke(
                "GetExecution", {"invocation_id": "transient-runtime-read"}
            )
            self.assertEqual(reconciled["status"], ExecutionStatus.RUNNING.value)
            self.assertNotEqual(
                reconciled["phase"], "command_status_temporarily_unavailable"
            )
        finally:
            service.close()
            sdk.close()

    def test_carrying_navigation_only_stops_after_sustained_unstable_load(self) -> None:
        sdk = self._sdk()
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        held = {
            "object_ref": "object://box-1",
            "tool_refs": [
                "component://tool/left",
                "component://tool/right",
            ],
        }
        state = sdk.state.snapshot()
        unsafe = ToolLoadAssessment(
            engaged=False,
            slipping=True,
            overloaded=False,
            sensor_fault=False,
            reasons=("component://tool/left:relative_motion_high",),
            tool_states=dict(state.tool_states),
        )
        safe = ToolLoadAssessment(
            engaged=True,
            slipping=False,
            overloaded=False,
            sensor_fault=False,
            reasons=(),
            tool_states=dict(state.tool_states),
        )
        try:
            route_ref = self._plan(service)
            request = {
                **self._route_request(route_ref, "carry-transient"),
                "navigation_purpose": "carry_to_place",
                "carrying_object": held,
            }
            service.invoke("FollowRoute", request)
            with patch(
                "r1pro_abilities.handlers.navigation.read_tool_load",
                side_effect=[(state, unsafe), (state, safe)],
            ):
                transient = service.invoke(
                    "GetExecution", {"invocation_id": "carry-transient"}
                )
                recovered = service.invoke(
                    "GetExecution", {"invocation_id": "carry-transient"}
                )
            self.assertEqual(transient["status"], ExecutionStatus.RUNNING.value)
            self.assertNotIn("_load_unstable_since", transient["result"])
            load_observation = next(
                item
                for item in transient["observations"]
                if item["kind"] == "navigation.carrying_load"
            )
            self.assertIn(
                "component://tool/left", load_observation["value"]["tool_states"]
            )
            self.assertIn(
                "hook_tangential_speed_m_s",
                load_observation["value"]["tool_states"]["component://tool/left"],
            )
            self.assertEqual(recovered["status"], ExecutionStatus.RUNNING.value)
            service.invoke(
                "StopExecution",
                {"invocation_id": "carry-transient", "reason": "test cleanup"},
            )

            route_ref = self._plan(service, "plan-persistent-load")
            request = {
                **self._route_request(route_ref, "carry-persistent"),
                "navigation_purpose": "carry_to_place",
                "carrying_object": held,
            }
            service.invoke("FollowRoute", request)
            with patch(
                "r1pro_abilities.handlers.navigation.read_tool_load",
                return_value=(state, unsafe),
            ):
                first = service.invoke(
                    "GetExecution", {"invocation_id": "carry-persistent"}
                )
                self.assertEqual(first["status"], ExecutionStatus.RUNNING.value)
                time.sleep(0.14)
                stopped = service.invoke(
                    "GetExecution", {"invocation_id": "carry-persistent"}
                )
            self.assertEqual(stopped["status"], ExecutionStatus.STOPPED.value)
            self.assertTrue(stopped["result"]["stop_evidence"]["holding"])
        finally:
            service.close()
            sdk.close()

    def test_tote_candidates_use_reachable_points_inside_side_grooves(self) -> None:
        sdk = self._sdk()
        try:
            tools = {
                item.side: item
                for item in sdk.state.capabilities().tools
                if item.kind == "tote_clamp"
            }
            robot_state = sdk.state.snapshot().model_copy(
                update={
                    "base_pose": Pose(
                        frame_id="world",
                        position=(0.0, 0.0, 0.02),
                        quaternion_xyzw=(0.0, 0.0, 2**-0.5, 2**-0.5),
                    )
                }
            )
            candidate = _candidate(
                "left_extract_first",
                1,
                {
                    "object_ref": "object://box-1",
                    "target_pose": {
                        "frame_id": "world",
                        # SceneSnapshot 对外返回箱体几何中心，而不是 MJCF body 原点。
                        "position_m": [0.0, 1.5, 0.32],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "object_extent_m": [0.6, 0.4, 0.34],
                    "target_revision": "scene-r1",
                },
                tools,
                {
                    "groove_axis_local": [1.0, 0.0, 0.0],
                    "groove_width_m": 0.230,
                    "tool_lateral_envelope_m": 0.052,
                    "wall_thickness_m": 0.004,
                    "groove_top_offset_m": 0.086,
                    "approach_clearance_m": 0.12,
                    "tool_access_envelope_m": 0.028,
                    "transfer_clearance_above_top_m": 0.02,
                    "groove_depth_m": 0.026,
                    "groove_backing_clearance_m": 0.002,
                    "hook_foot_length_m": 0.028,
                    "hook_forward_reach_from_load_frame_m": 0.017,
                    "hook_upper_reach_from_load_frame_m": 0.012,
                    "hook_foot_thickness_m": 0.006,
                    "hook_engagement_travel_m": 0.008,
                    "clamp_target_position_m": 0.0,
                    "maximum_extraction_distance_m": 0.18,
                    "lift_distance_m": 0.16,
                },
                robot_state,
            )
            self.assertEqual(
                [item["target_position_m"] for item in candidate["clamp_setpoints"]],
                [0.0, 0.0],
            )
            transfer = {
                item["tool_ref"]: item["target_pose"]
                for item in candidate["transfer_poses"]
            }
            clearance = {
                item["tool_ref"]: item["target_pose"]
                for item in candidate["clearance_poses"]
            }
            self.assertEqual(
                clearance["component://tool/left"]["position_m"],
                [-0.42, 1.18, 0.51],
            )
            self.assertEqual(
                clearance["component://tool/left"]["orientation_xyzw"],
                list(robot_state.end_effectors["left"].quaternion_xyzw),
            )
            approach = {
                item["tool_ref"]: item["target_pose"]
                for item in candidate["approach_poses"]
            }
            insert = {
                item["tool_ref"]: item["target_pose"]
                for item in candidate["hook_insert_poses"]
            }
            # 两侧都抓连续侧面凹槽中靠Robot的一段；52mm夹具实体完整
            # 位于230mm槽内，并在前端保留4mm几何余量。
            self.assertEqual(
                approach["component://tool/left"]["position_m"],
                [-0.42, 1.415, 0.384],
            )
            self.assertEqual(
                approach["component://tool/right"]["position_m"],
                [0.42, 1.415, 0.384],
            )
            # load frame进入7mm后，钩尖在凹槽后壁前保留2mm间隙。
            self.assertEqual(
                insert["component://tool/left"]["position_m"],
                [-0.293, 1.415, 0.384],
            )
            self.assertEqual(
                insert["component://tool/right"]["position_m"],
                [0.293, 1.415, 0.384],
            )
            self.assertEqual(
                transfer["component://tool/left"]["position_m"],
                [-0.42, 1.415, 0.51],
            )
            self.assertEqual(
                transfer["component://tool/right"]["position_m"],
                [0.42, 1.415, 0.51],
            )
            self.assertEqual(
                approach["component://tool/left"]["orientation_xyzw"],
                [0.0, 0.0, 0.0, 1.0],
            )
            right_orientation = approach["component://tool/right"]["orientation_xyzw"]
            self.assertAlmostEqual(abs(right_orientation[2]), 1.0)
            self.assertAlmostEqual(right_orientation[3], 0.0)
            seat_position = candidate["hook_seat_poses"][0]["target_pose"]["position_m"]
            self.assertAlmostEqual(seat_position[2], 0.392)
            self.assertAlmostEqual(
                seat_position[2] - insert["component://tool/left"]["position_m"][2],
                0.008,
            )
            # 没有同层邻箱时，显式选择单侧候选也不应制造无意义外拉。
            self.assertEqual(candidate["pull_path"], [])
        finally:
            sdk.close()

    def test_dense_layer_prefers_external_extraction_side(self) -> None:
        target_pose = {
            "frame_id": "world",
            "position_m": [-0.31, 1.29, 1.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        target = SceneObject(
            source_id="tote-target",
            category="tote",
            name="target",
            pose=Pose(
                frame_id="world",
                position=(-0.31, 1.29, 1.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            extent=(0.6, 0.4, 0.34),
        )
        inner_neighbor = SceneObject(
            source_id="tote-neighbor",
            category="tote",
            name="neighbor",
            pose=Pose(
                frame_id="world",
                position=(0.31, 1.29, 1.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            extent=(0.6, 0.4, 0.34),
        )
        ordered = _order_strategies_for_scene(
            list(("direct_bilateral", "left_extract_first", "right_extract_first")),
            {
                "object_ref": "tote-target",
                "target_pose": target_pose,
                "object_extent_m": [0.6, 0.4, 0.34],
            },
            {"groove_axis_local": [1.0, 0.0, 0.0], "approach_clearance_m": 0.12},
            Pose(
                frame_id="world",
                position=(-0.31, 0.5, 0.02),
                quaternion_xyzw=(0.0, 0.0, 2**-0.5, 2**-0.5),
            ),
            [target, inner_neighbor],
        )
        self.assertEqual(
            ordered,
            ["left_extract_first", "direct_bilateral", "right_extract_first"],
        )

    def test_dense_layer_extraction_uses_one_clearance_driven_pull(self) -> None:
        """首轮按实时通道缺口一次主拉取，不再拆成毫米级循环。"""

        sdk = self._sdk()
        try:
            tools = {
                item.side: item
                for item in sdk.state.capabilities().tools
                if item.kind == "tote_clamp"
            }
            robot_state = sdk.state.snapshot().model_copy(
                update={
                    "base_pose": Pose(
                        frame_id="world",
                        position=(-0.31, 0.5, 0.02),
                        quaternion_xyzw=(0.0, 0.0, 2**-0.5, 2**-0.5),
                    )
                }
            )
            target = SceneObject(
                source_id="tote-target",
                category="tote",
                name="target",
                pose=Pose(
                    frame_id="world",
                    position=(-0.31, 1.29, 1.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                extent=(0.6, 0.4, 0.34),
            )
            neighbor = SceneObject(
                source_id="tote-neighbor",
                category="tote",
                name="neighbor",
                pose=Pose(
                    frame_id="world",
                    position=(0.31, 1.29, 1.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                extent=(0.6, 0.4, 0.34),
            )
            other_row = SceneObject(
                source_id="tote-other-row",
                category="tote",
                name="other-row",
                pose=Pose(
                    frame_id="world",
                    position=(-0.38, 2.0, 1.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                extent=(0.6, 0.4, 0.34),
            )
            settings = {
                "groove_axis_local": [1.0, 0.0, 0.0],
                "groove_top_offset_m": 0.086,
                "approach_clearance_m": 0.12,
                "tool_access_envelope_m": 0.028,
                "groove_depth_m": 0.026,
                "groove_backing_clearance_m": 0.002,
                "hook_foot_length_m": 0.028,
                "hook_forward_reach_from_load_frame_m": 0.017,
                "hook_upper_reach_from_load_frame_m": 0.012,
                "hook_foot_thickness_m": 0.006,
                "hook_engagement_travel_m": 0.008,
                "maximum_extraction_distance_m": 0.18,
            }
            candidate = _candidate(
                "left_extract_first",
                1,
                {
                    "object_ref": "tote-target",
                    "target_pose": {
                        "frame_id": "world",
                        "position_m": [-0.31, 1.29, 1.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "object_extent_m": [0.6, 0.4, 0.34],
                    "target_revision": "scene-r1",
                },
                tools,
                settings,
                robot_state,
                [target, neighbor, other_row],
            )

            commanded_extraction = candidate["pull_distance_m"]
            self.assertEqual(candidate["pull_direction_world"], [-1.0, 0.0, 0.0])
            initial_surface_gap = 0.62 - 0.6
            path_clearance = settings["approach_clearance_m"]
            self.assertAlmostEqual(
                commanded_extraction,
                max(0.0, path_clearance - initial_surface_gap),
            )
            pull_z = candidate["pull_path"][0]["target_pose"]["position_m"][2]
            expected_groove_height = (
                target.pose.position[2]
                + target.extent[2] / 2.0
                - settings["groove_top_offset_m"]
                - settings["hook_upper_reach_from_load_frame_m"]
            )
            self.assertAlmostEqual(pull_z, expected_groove_height)
        finally:
            sdk.close()

    def test_engaged_side_pull_starts_from_current_contact_pose(self) -> None:
        """接触后的外拉从实时Pose开始，并保持当前侧面凹槽高度。"""

        sdk = self._sdk()
        try:
            tools = {
                item.side: item
                for item in sdk.state.capabilities().tools
                if item.kind == "tote_clamp"
            }
            snapshot = sdk.state.snapshot()
            current_left = Pose(
                frame_id="world",
                position=(-0.590, 1.242, 1.070),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
            robot_state = snapshot.model_copy(
                update={
                    "base_pose": Pose(
                        frame_id="world",
                        position=(-0.31, 0.5, 0.02),
                        quaternion_xyzw=(0.0, 0.0, 2**-0.5, 2**-0.5),
                    ),
                    "end_effectors": {
                        **snapshot.end_effectors,
                        "left": current_left,
                    },
                }
            )
            target = SceneObject(
                source_id="tote-target",
                category="tote",
                name="target",
                pose=Pose(
                    frame_id="world",
                    position=(-0.31, 1.29, 1.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                extent=(0.6, 0.4, 0.34),
            )
            neighbor = SceneObject(
                source_id="tote-neighbor",
                category="tote",
                name="neighbor",
                pose=Pose(
                    frame_id="world",
                    position=(0.31, 1.29, 1.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                extent=(0.6, 0.4, 0.34),
            )
            insertion = 0.022
            candidate = _candidate(
                "left_extract_first",
                1,
                {
                    "object_ref": "tote-target",
                    "target_pose": {
                        "frame_id": "world",
                        "position_m": [-0.31, 1.29, 1.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "object_extent_m": [0.6, 0.4, 0.34],
                    "target_revision": "scene-contact",
                    "engaged_tool_ref": "component://tool/left",
                },
                tools,
                {
                    "groove_axis_local": [1.0, 0.0, 0.0],
                    "groove_top_offset_m": 0.086,
                    "approach_clearance_m": 0.12,
                    "tool_access_envelope_m": 0.028,
                    "groove_depth_m": insertion + 0.019,
                    "groove_backing_clearance_m": 0.002,
                    "hook_foot_length_m": 0.028,
                    "hook_forward_reach_from_load_frame_m": 0.017,
                    "hook_upper_reach_from_load_frame_m": 0.012,
                    "hook_foot_thickness_m": 0.006,
                    "hook_engagement_travel_m": 0.008,
                    "maximum_extraction_distance_m": 0.18,
                    "maximum_extraction_step_m": 0.06,
                },
                robot_state,
                [target, neighbor],
            )

            pull = candidate["pull_path"][0]["target_pose"]
            initial_gap = 0.62 - 0.6
            # 首侧接合后执行厘米级连续主拉，但不让单次命令耗尽凹槽接合
            # 余量；每步结束后以实时物体位移和完整右臂路径决定是否继续。
            step = min(0.12 - initial_gap, 0.06)
            self.assertAlmostEqual(
                pull["position_m"][0], current_left.position[0] - step
            )
            self.assertAlmostEqual(
                pull["position_m"][2],
                current_left.position[2],
            )
        finally:
            sdk.close()

    def test_engaged_side_channel_uses_complete_approach_corridor(self) -> None:
        """完整前臂通道满足后才进入路径预检，不能只看钩脚宽度。"""

        sdk = self._sdk()
        try:
            tools = {
                item.side: item
                for item in sdk.state.capabilities().tools
                if item.kind == "tote_clamp"
            }
            snapshot = sdk.state.snapshot()
            robot_state = snapshot.model_copy(
                update={
                    "base_pose": Pose(
                        frame_id="world",
                        position=(-0.31, 0.5, 0.02),
                        quaternion_xyzw=(0.0, 0.0, 2**-0.5, 2**-0.5),
                    )
                }
            )
            target = SceneObject(
                source_id="tote-target",
                category="tote",
                name="target",
                pose=Pose(
                    frame_id="world",
                    position=(-0.41, 1.29, 1.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                extent=(0.6, 0.4, 0.34),
            )
            neighbor = SceneObject(
                source_id="tote-neighbor",
                category="tote",
                name="neighbor",
                pose=Pose(
                    frame_id="world",
                    position=(0.31, 1.29, 1.0),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                extent=(0.6, 0.4, 0.34),
            )
            candidate = _candidate(
                "left_extract_first",
                1,
                {
                    "object_ref": "tote-target",
                    "target_pose": {
                        "frame_id": "world",
                        "position_m": list(target.pose.position),
                        "orientation_xyzw": list(target.pose.quaternion_xyzw),
                    },
                    "object_extent_m": list(target.extent),
                    "target_revision": "scene-final-gap",
                    "engaged_tool_ref": "component://tool/left",
                },
                tools,
                {
                    "groove_axis_local": [1.0, 0.0, 0.0],
                    "groove_top_offset_m": 0.086,
                    "approach_clearance_m": 0.12,
                    "tool_access_envelope_m": 0.028,
                    "groove_depth_m": 0.035,
                    "groove_backing_clearance_m": 0.002,
                    "hook_foot_length_m": 0.028,
                    "hook_forward_reach_from_load_frame_m": 0.017,
                    "hook_upper_reach_from_load_frame_m": 0.012,
                    "hook_foot_thickness_m": 0.006,
                    "hook_engagement_travel_m": 0.008,
                    "maximum_extraction_distance_m": 0.18,
                    "maximum_extraction_step_m": 0.012,
                },
                robot_state,
                [target, neighbor],
            )

            self.assertEqual(candidate["_secondary_preflight_mode"], "path")
            expected_approach_offset = 0.028
            right_approach = next(
                item["target_pose"]
                for item in candidate["approach_poses"]
                if item["tool_ref"] == "component://tool/right"
            )
            target_right_surface_x = target.pose.position[0] + target.extent[0] / 2.0
            self.assertAlmostEqual(
                right_approach["position_m"][0] - target_right_surface_x,
                expected_approach_offset,
                places=6,
            )
        finally:
            sdk.close()

    def test_engaged_side_forms_work_corridor_before_secondary_path(self) -> None:
        """首侧接合后先形成厘米级工作通道，再验证完整右臂路径。"""

        sdk = self._sdk()
        try:
            tools = {
                item.side: item
                for item in sdk.state.capabilities().tools
                if item.kind == "tote_clamp"
            }
            robot_state = sdk.state.snapshot().model_copy(
                update={
                    "base_pose": Pose(
                        frame_id="world",
                        position=(-0.31, 0.5, 0.02),
                        quaternion_xyzw=(0.0, 0.0, 2**-0.5, 2**-0.5),
                    )
                }
            )
            target = SceneObject(
                source_id="tote-target",
                category="tote",
                name="target",
                pose=Pose(
                    frame_id="world",
                    position=(-0.42, 1.29, 1.0),
                    quaternion_xyzw=(0.0, 0.0, 0.1305262, 0.9914449),
                ),
                extent=(0.6, 0.4, 0.34),
            )
            neighbor = SceneObject(
                source_id="tote-neighbor",
                category="tote",
                name="neighbor",
                pose=Pose(
                    frame_id="world",
                    position=(0.22, 1.29, 1.0),
                    quaternion_xyzw=(0.0, 0.0, 0.1305262, 0.9914449),
                ),
                extent=(0.6, 0.4, 0.34),
            )
            candidate = _candidate(
                "left_extract_first",
                1,
                {
                    "object_ref": "tote-target",
                    "target_pose": {
                        "frame_id": "world",
                        "position_m": [-0.42, 1.29, 1.0],
                        "orientation_xyzw": [0.0, 0.0, 0.1305262, 0.9914449],
                    },
                    "object_extent_m": [0.6, 0.4, 0.34],
                    "target_revision": "scene-r2",
                    "engaged_tool_ref": "component://tool/left",
                },
                tools,
                {
                    "groove_axis_local": [1.0, 0.0, 0.0],
                    "groove_top_offset_m": 0.086,
                    "approach_clearance_m": 0.12,
                    "tool_access_envelope_m": 0.028,
                    "groove_depth_m": 0.026,
                    "groove_backing_clearance_m": 0.002,
                    "hook_foot_length_m": 0.028,
                    "hook_forward_reach_from_load_frame_m": 0.017,
                    "hook_upper_reach_from_load_frame_m": 0.012,
                    "hook_foot_thickness_m": 0.006,
                    "hook_engagement_travel_m": 0.008,
                    "maximum_extraction_distance_m": 0.18,
                },
                robot_state,
                [target, neighbor],
            )

            # 已接合左侧后，右侧load frame紧贴目标槽口外侧；新增通道
            # 全部留给右腕和前臂，不能再把工具参考点居中推向邻箱。
            target_pose = target.pose
            groove_axis = (
                math.cos(math.radians(15.0)),
                math.sin(math.radians(15.0)),
                0.0,
            )
            # 两个箱体使用相同15度姿态，沿自身凹槽轴的真实跨度就是
            # 局部X尺寸；不能把世界轴再乘一次局部长宽。
            target_axis_span = target.extent[0]
            neighbor_axis_span = neighbor.extent[0]
            neighbor_delta = tuple(
                neighbor.pose.position[index] - target_pose.position[index]
                for index in range(3)
            )
            surface_gap = (
                abs(
                    sum(
                        neighbor_delta[index] * groove_axis[index] for index in range(3)
                    )
                )
                - (target_axis_span + neighbor_axis_span) / 2.0
            )
            right_approach = next(
                item["target_pose"]
                for item in candidate["approach_poses"]
                if item["tool_ref"] == "component://tool/right"
            )
            approach_delta = tuple(
                right_approach["position_m"][index] - target_pose.position[index]
                for index in range(3)
            )
            radial_distance = (
                abs(
                    sum(
                        approach_delta[index] * groove_axis[index] for index in range(3)
                    )
                )
                - target.extent[0] / 2.0
            )
            expected_offset = 0.028
            self.assertAlmostEqual(radial_distance, expected_offset)
            right_clearance = next(
                item["target_pose"]
                for item in candidate["clearance_poses"]
                if item["tool_ref"] == "component://tool/right"
            )
            right_transfer = next(
                item["target_pose"]
                for item in candidate["transfer_poses"]
                if item["tool_ref"] == "component://tool/right"
            )
            right_alignment = next(
                item["target_pose"]
                for item in candidate["alignment_poses"]
                if item["tool_ref"] == "component://tool/right"
            )
            transfer_delta = tuple(
                right_alignment["position_m"][index] - target_pose.position[index]
                for index in range(3)
            )
            transfer_radial_distance = (
                abs(
                    sum(
                        transfer_delta[index] * groove_axis[index] for index in range(3)
                    )
                )
                - target.extent[0] / 2.0
            )
            self.assertAlmostEqual(
                transfer_radial_distance,
                expected_offset,
            )
            # layout001成功链路中，clearance只前展并抬高一个工具净空；
            # transfer进入箱前安全高位，alignment只保留最后一个短段。
            current_right = robot_state.end_effectors["right"]
            self.assertAlmostEqual(
                math.dist(
                    current_right.position[:2],
                    right_clearance["position_m"][:2],
                ),
                0.12,
            )
            self.assertAlmostEqual(
                right_clearance["position_m"][2], current_right.position[2] + 0.12
            )
            self.assertEqual(
                right_clearance["orientation_xyzw"], right_transfer["orientation_xyzw"]
            )
            self.assertGreater(
                math.dist(
                    current_right.position[:2],
                    right_transfer["position_m"][:2],
                ),
                math.dist(
                    current_right.position[:2],
                    right_clearance["position_m"][:2],
                ),
            )
            self.assertGreater(
                right_transfer["position_m"][2],
                right_clearance["position_m"][2],
            )
            self.assertLessEqual(
                math.dist(
                    right_transfer["position_m"][:2],
                    right_alignment["position_m"][:2],
                ),
                0.12 + 0.028 + 1e-9,
            )
            self.assertAlmostEqual(
                right_alignment["position_m"][2],
                right_transfer["position_m"][2],
            )
            mirrored_candidate = _candidate(
                "right_extract_first",
                2,
                {
                    "object_ref": "tote-target",
                    "target_pose": {
                        "frame_id": "world",
                        "position_m": [-0.42, 1.29, 1.0],
                        "orientation_xyzw": [0.0, 0.0, 0.1305262, 0.9914449],
                    },
                    "object_extent_m": [0.6, 0.4, 0.34],
                    "target_revision": "scene-r2",
                    "engaged_tool_ref": "component://tool/right",
                },
                tools,
                {
                    "groove_axis_local": [1.0, 0.0, 0.0],
                    "groove_top_offset_m": 0.086,
                    "approach_clearance_m": 0.12,
                    "tool_access_envelope_m": 0.028,
                    "groove_depth_m": 0.026,
                    "groove_backing_clearance_m": 0.0015,
                    "hook_foot_length_m": 0.028,
                    "hook_forward_reach_from_load_frame_m": 0.017,
                    "hook_upper_reach_from_load_frame_m": 0.012,
                    "hook_foot_thickness_m": 0.006,
                    "hook_engagement_travel_m": 0.008,
                    "maximum_extraction_distance_m": 0.18,
                },
                robot_state,
                [target, neighbor],
            )
            left_transfer = next(
                item["target_pose"]
                for item in mirrored_candidate["transfer_poses"]
                if item["tool_ref"] == "component://tool/left"
            )
            left_alignment = next(
                item["target_pose"]
                for item in mirrored_candidate["alignment_poses"]
                if item["tool_ref"] == "component://tool/left"
            )
            self.assertAlmostEqual(
                left_alignment["position_m"][2] - left_transfer["position_m"][2],
                0.12,
            )
            self.assertEqual(expected_offset, 0.028)
            self.assertLess(surface_gap, 0.028)
            self.assertEqual(
                candidate["_secondary_preflight_mode"], "additional_extraction"
            )
            with (
                patch(
                    "r1pro_abilities.handlers.grasp_planning.scene_collision_environment",
                    return_value=None,
                ),
                patch(
                    "r1pro_abilities.handlers.grasp_planning._preflight_candidate_groups",
                    side_effect=[
                        (True, None),
                        (True, None),
                    ],
                ) as preflight,
            ):
                reachable, reason = _candidate_reachability(
                    sdk,
                    candidate,
                    robot_state,
                    tools,
                    engaged_tool_ref="component://tool/left",
                    object_ref="tote-target",
                )
            self.assertTrue(reachable, reason)
            self.assertEqual(candidate["strategy"], "left_extract_first")
            self.assertTrue(candidate["pull_path"])
            pull_target = candidate["pull_path"][0]["target_pose"]["position_m"]
            current_left = robot_state.end_effectors["left"].position
            commanded_distance = math.dist(pull_target, current_left)
            self.assertGreater(commanded_distance, 0.0)
            self.assertAlmostEqual(
                commanded_distance,
                min(0.18, max(0.0, 0.12 - surface_gap)),
            )
            self.assertEqual(preflight.call_count, 1)
            self.assertFalse(preflight.call_args.kwargs["validate_motion_paths"])
        finally:
            sdk.close()

    def test_transport_posture_preserves_observed_tool_object_relationship(
        self,
    ) -> None:
        """携物目标应平移真实抓取关系，而不是复用抓取前理想seat。"""

        sdk = self._sdk()
        models = ModelProfileRegistry.load(ROOT / "configs/models.example.json")
        service = R1ProAbilityService(AbilityRole.GRASP_PLANNING, sdk, models)
        state = sdk.state.snapshot()
        request = {
            "robot_id": "r1pro-fake-001",
            "object_ref": "object://box-1",
            "object_pose": {
                "frame_id": state.end_effectors["left"].frame_id,
                "position_m": [0.8, 0.2, 0.4],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "revision": "held-r1",
            },
            "object_extent_m": [0.6, 0.4, 0.34],
            "target_revision": "held-r1",
            "tool_refs": [
                "component://tool/left",
                "component://tool/right",
            ],
        }
        scene = SceneSnapshot(
            scene_key="depalletizing",
            instance_id="scene-transport",
            generation=1,
            coordinate_frame="world",
            objects=[
                SceneObject(
                    source_id="object://box-1",
                    category="tote",
                    name="target",
                    pose=Pose(
                        frame_id="world",
                        position=(0.8, 0.2, 0.4),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(0.6, 0.4, 0.34),
                ),
                SceneObject(
                    source_id="object://neighbor",
                    category="tote",
                    name="neighbor",
                    pose=Pose(
                        frame_id="world",
                        position=(0.8, 0.0, 0.4),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(0.6, 0.4, 0.34),
                ),
            ],
            regions=[],
            observed_at=datetime.fromisoformat("2026-08-20T00:00:00+00:00"),
        )
        scene_patch = patch.object(sdk.state, "scene_snapshot", return_value=scene)
        scene_patch.start()
        try:
            result = service.invoke("PlanTransportPosture", request)
            self.assertEqual(result["status"], ExecutionStatus.SUCCEEDED.value)
            poses = {
                item["tool_ref"]: item["target_pose"]
                for item in result["result"]["transport_poses"]
            }
            clearance_poses = {
                item["tool_ref"]: item["target_pose"]
                for item in result["result"]["clearance_poses"]
            }
            deltas = []
            clearance_deltas = []
            for side in ("left", "right"):
                tool_ref = f"component://tool/{side}"
                current = state.end_effectors[side]
                target = poses[tool_ref]
                deltas.append(
                    tuple(
                        target["position_m"][index] - current.position[index]
                        for index in range(3)
                    )
                )
                clearance_target = clearance_poses[tool_ref]
                clearance_deltas.append(
                    tuple(
                        clearance_target["position_m"][index] - current.position[index]
                        for index in range(3)
                    )
                )
                self.assertEqual(
                    target["orientation_xyzw"],
                    list(current.quaternion_xyzw),
                )
            self.assertEqual(deltas[0], deltas[1])
            self.assertEqual(clearance_deltas[0], clearance_deltas[1])
            self.assertAlmostEqual(clearance_deltas[0][0], -0.6)
            self.assertAlmostEqual(clearance_deltas[0][1], 0.0)
            self.assertAlmostEqual(deltas[0][0], -0.25)
            self.assertAlmostEqual(deltas[0][1], 0.2)
            # 同层邻箱位于横向居中扫掠区时，先按实时AABB退到两箱前后
            # 包络之外；最终姿态恢复Profile的0.55m前向携物距离，并向
            # 当前无障碍侧保留0.4m横向位置，不能把箱体重新推回邻箱。
            self.assertAlmostEqual(
                result["result"]["desired_object_pose"]["position_m"][0],
                0.55,
            )
            self.assertAlmostEqual(
                result["result"]["desired_object_pose"]["position_m"][1],
                0.4,
            )

            request["object_pose"]["position_m"][2] = 1.08
            high_result = service.invoke("PlanTransportPosture", request)
            self.assertEqual(high_result["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertAlmostEqual(
                high_result["result"]["desired_object_pose"]["position_m"][2],
                1.08,
            )
        finally:
            scene_patch.stop()
            service.close()
            sdk.close()

    def test_grasp_planning_filters_unreachable_bilateral_candidate(self) -> None:
        sdk = self._sdk()
        models = ModelProfileRegistry.load(ROOT / "configs/models.example.json")
        service = R1ProAbilityService(AbilityRole.GRASP_PLANNING, sdk, models)

        def reject_direct(_sdk, candidate, *_args, **_kwargs):
            if candidate["strategy"] == "direct_bilateral":
                return False, "direct_bilateral/bilateral_transfer: 双侧目标不可达"
            return True, None

        try:
            with patch(
                "r1pro_abilities.handlers.grasp_planning._preflight_candidate_groups",
                side_effect=reject_direct,
            ):
                result = service.invoke(
                    "GenerateCandidates", self._grasp_planning_request()
                )
            self.assertEqual(result["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertEqual(
                [item["strategy"] for item in result["result"]["candidates"]],
                ["left_extract_first", "right_extract_first"],
            )
            self.assertEqual(
                result["result"]["rejected_candidates"],
                ["direct_bilateral/bilateral_transfer: 双侧目标不可达"],
            )
        finally:
            service.close()
            sdk.close()

    def test_grasp_planning_preflights_only_contact_free_entry(self) -> None:
        sdk = self._sdk()
        models = ModelProfileRegistry.load(ROOT / "configs/models.example.json")
        service = R1ProAbilityService(AbilityRole.GRASP_PLANNING, sdk, models)
        try:
            with patch(
                "r1pro_abilities.handlers.grasp_planning._preflight_candidate_groups",
                return_value=(True, None),
            ) as preflight:
                result = service.invoke(
                    "GenerateCandidates", self._grasp_planning_request()
                )
            self.assertEqual(result["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertEqual(len(result["result"]["candidates"]), 3)
            self.assertEqual(preflight.call_count, 3)
            phases_by_strategy = {
                call.args[1]["strategy"]: [phase for phase, _group in call.args[4]]
                for call in preflight.call_args_list
            }
            self.assertEqual(
                phases_by_strategy["direct_bilateral"],
                ["bilateral_clearance"],
            )
            for strategy in ("left_extract_first", "right_extract_first"):
                self.assertEqual(
                    phases_by_strategy[strategy],
                    ["first_primary_clearance"],
                )
                call = next(
                    value
                    for value in preflight.call_args_list
                    if value.args[1]["strategy"] == strategy
                )
                groups = dict(call.args[4])
                # 候选生成只确认首个clearance目标；后续动作从真实终点规划。
                primary_ref = (
                    "component://tool/left"
                    if strategy == "left_extract_first"
                    else "component://tool/right"
                )
                for phase in ("first_primary_clearance",):
                    self.assertEqual(len(groups[phase]), 1)
                    self.assertEqual(groups[phase][0]["tool_ref"], primary_ref)
            self.assertFalse(
                preflight.call_args_list[0].kwargs["validate_motion_paths"]
            )
            self.assertFalse(
                preflight.call_args_list[1].kwargs["validate_motion_paths"]
            )
            self.assertFalse(
                preflight.call_args_list[2].kwargs["validate_motion_paths"]
            )
        finally:
            service.close()
            sdk.close()

    def test_grasp_planning_pulls_only_for_real_entry_gap(self) -> None:
        """右臂clearance可达且入口净空不足时才执行一个短拉。"""

        sdk = self._sdk()
        state = sdk.state.snapshot()
        request = self._grasp_planning_request()
        tools = {
            item.side: item
            for item in sdk.state.capabilities().tools
            if item.kind == "tote_clamp"
        }
        profile = ModelProfileRegistry.load(
            ROOT / "configs/models.example.json"
        ).resolve("grasp_planning", None)
        try:
            candidate = _candidate(
                "left_extract_first", 0, request, tools, profile.settings, state
            )
            current_left = state.end_effectors["left"]
            candidate["_secondary_preflight_mode"] = "additional_extraction"
            candidate["pull_path"] = [
                {
                    "tool_ref": "component://tool/left",
                    "target_pose": {
                        "frame_id": current_left.frame_id,
                        "position_m": [
                            current_left.position[0] + 0.001,
                            current_left.position[1],
                            current_left.position[2],
                        ],
                        "orientation_xyzw": list(current_left.quaternion_xyzw),
                    },
                }
            ]
            with (
                patch(
                    "r1pro_abilities.handlers.grasp_planning.scene_collision_environment",
                    return_value=None,
                ),
                patch(
                    "r1pro_abilities.handlers.grasp_planning._preflight_candidate_groups",
                    side_effect=[
                        (True, None),
                        (True, None),
                    ],
                ) as preflight,
            ):
                reachable, reason = _candidate_reachability(
                    sdk,
                    candidate,
                    state,
                    tools,
                    engaged_tool_ref="component://tool/left",
                    object_ref=str(request["object_ref"]),
                )
            self.assertTrue(reachable, reason)
            self.assertTrue(candidate["pull_path"])
            self.assertEqual(candidate["strategy"], "left_extract_first")
            self.assertEqual(preflight.call_count, 1)
            self.assertNotIn(
                "allow_shared_joint_compensation",
                preflight.call_args.kwargs,
            )
            self.assertFalse(preflight.call_args.kwargs["validate_motion_paths"])

            candidate["_secondary_preflight_mode"] = "path"
            candidate["pull_path"] = []
            with (
                patch(
                    "r1pro_abilities.handlers.grasp_planning.scene_collision_environment",
                    return_value=None,
                ),
                patch(
                    "r1pro_abilities.handlers.grasp_planning._preflight_candidate_groups",
                    return_value=(True, None),
                ) as ready_preflight,
            ):
                reachable, reason = _candidate_reachability(
                    sdk,
                    candidate,
                    state,
                    tools,
                    engaged_tool_ref="component://tool/left",
                    object_ref=str(request["object_ref"]),
                )
            self.assertTrue(reachable, reason)
            self.assertEqual(candidate["strategy"], "direct_bilateral")
            self.assertEqual(candidate["pull_path"], [])
            self.assertEqual(ready_preflight.call_count, 1)
            self.assertTrue(ready_preflight.call_args.kwargs["validate_motion_paths"])
            ready_groups = ready_preflight.call_args.args[4]
            self.assertEqual(
                [name for name, _items in ready_groups],
                ["secondary_clearance"],
                "接触后候选只应预检下一段，不能提前否决未来alignment",
            )
            self.assertEqual(
                ready_preflight.call_args.args[5],
                {},
                "外拉后的clearance应先使用第二侧局部解，首侧由实时接触监控",
            )
            candidate["strategy"] = "left_extract_first"
            candidate["candidate_id"] = "left_extract_first-0"
            candidate["_secondary_preflight_mode"] = "path"
            with (
                patch(
                    "r1pro_abilities.handlers.grasp_planning.scene_collision_environment",
                    return_value=None,
                ),
                patch(
                    "r1pro_abilities.handlers.grasp_planning._preflight_candidate_groups",
                    return_value=(True, None),
                ) as resumed_preflight,
            ):
                reachable, reason = _candidate_reachability(
                    sdk,
                    candidate,
                    state,
                    tools,
                    engaged_tool_ref="component://tool/left",
                    secondary_resume_phase="insert",
                    object_ref=str(request["object_ref"]),
                )
            self.assertTrue(reachable, reason)
            self.assertEqual(resumed_preflight.call_count, 1)
            self.assertEqual(
                resumed_preflight.call_args.args[4],
                [],
                "第二手到达槽口外后不能重新预检已完成的clearance路径",
            )
        finally:
            sdk.close()

    def test_secondary_insert_refresh_preserves_current_continuous_groove_lane(
        self,
    ) -> None:
        """重观测只修正径向插入，不把已到槽口的第二手横移到另一个抓取点。"""

        sdk = self._sdk()
        try:
            state = sdk.state.snapshot()
            current_right = state.end_effectors["right"]
            desired_lane_offset = 0.03
            right_at_slot = current_right.model_copy(
                update={
                    "position": (
                        current_right.position[0],
                        float(POSE["position_m"][1]) + desired_lane_offset,
                        current_right.position[2],
                    )
                }
            )
            state = state.model_copy(
                update={
                    "end_effectors": {
                        **state.end_effectors,
                        "right": right_at_slot,
                    }
                }
            )
            request = self._grasp_planning_request()
            request.update(
                {
                    "engaged_tool_ref": "component://tool/left",
                    "secondary_resume_phase": "insert",
                }
            )
            tools = {
                item.side: item
                for item in sdk.state.capabilities().tools
                if item.kind == "tote_clamp"
            }
            profile = ModelProfileRegistry.load(
                ROOT / "configs/models.example.json"
            ).resolve("grasp_planning", "mujoco-tote-grasp")
            candidate = _candidate(
                "left_extract_first",
                0,
                request,
                tools,
                profile.settings,
                state,
            )

            for key in ("pregrasp_poses", "hook_insert_poses", "hook_seat_poses"):
                right_target = next(
                    item["target_pose"]
                    for item in candidate[key]
                    if item["tool_ref"] == "component://tool/right"
                )
                self.assertAlmostEqual(
                    right_target["position_m"][1],
                    float(POSE["position_m"][1]) + desired_lane_offset,
                    places=6,
                )
        finally:
            sdk.close()

    def test_secondary_transfer_failure_does_not_rewrite_path(self) -> None:
        """IK失败必须保留原路径诊断，不能静默改写成另一条物理动作。"""

        sdk = self._sdk()
        state = sdk.state.snapshot()
        request = self._grasp_planning_request()
        request["engaged_tool_ref"] = "component://tool/left"
        tools = {
            item.side: item
            for item in sdk.state.capabilities().tools
            if item.kind == "tote_clamp"
        }
        profile = ModelProfileRegistry.load(
            ROOT / "configs/models.example.json"
        ).resolve("grasp_planning", None)
        candidate = _candidate(
            "left_extract_first",
            0,
            request,
            tools,
            profile.settings,
            state,
        )
        original_transfer = next(
            item["target_pose"]
            for item in candidate["transfer_poses"]
            if item["tool_ref"] == "component://tool/right"
        )
        try:
            with (
                patch(
                    "r1pro_abilities.handlers.grasp_planning.scene_collision_environment",
                    return_value=None,
                ),
                patch(
                    "r1pro_abilities.handlers.grasp_planning._preflight_candidate_groups",
                    return_value=(
                        False,
                        "left_extract_first/secondary_transfer: "
                        "Pinocchio 多末端IK 未在有限多起点内收敛",
                    ),
                ) as preflight,
            ):
                reachable, reason = _candidate_reachability(
                    sdk,
                    candidate,
                    state,
                    tools,
                    engaged_tool_ref="component://tool/left",
                    object_ref=str(request["object_ref"]),
                )

            self.assertFalse(reachable)
            self.assertIn("secondary_transfer", str(reason))
            self.assertEqual(preflight.call_count, 1)
            selected_transfer = next(
                item["target_pose"]
                for item in candidate["transfer_poses"]
                if item["tool_ref"] == "component://tool/right"
            )
            self.assertEqual(selected_transfer, original_transfer)
            self.assertTrue(preflight.call_args.kwargs["validate_motion_paths"])
        finally:
            sdk.close()

    def test_grasp_planning_reports_when_current_site_has_no_reachable_candidate(
        self,
    ) -> None:
        sdk = self._sdk()
        models = ModelProfileRegistry.load(ROOT / "configs/models.example.json")
        service = R1ProAbilityService(AbilityRole.GRASP_PLANNING, sdk, models)

        def reject_all(_sdk, candidate, *_args, **_kwargs):
            return (
                False,
                f"{candidate['strategy']}/contact_free_entry: 当前站位不可达",
            )

        try:
            with patch(
                "r1pro_abilities.handlers.grasp_planning._preflight_candidate_groups",
                side_effect=reject_all,
            ):
                result = service.invoke(
                    "GenerateCandidates", self._grasp_planning_request()
                )
            self.assertEqual(result["status"], ExecutionStatus.FAILED.value)
            self.assertEqual(result["error"]["code"], "GRASP_SITE_UNREACHABLE")
            self.assertIn(
                "direct_bilateral/contact_free_entry", result["error"]["message"]
            )
            self.assertIn("当前站位不可达", result["error"]["message"])
        finally:
            service.close()
            sdk.close()

    def test_insert_motion_forwards_contact_safety_without_early_completion(
        self,
    ) -> None:
        sdk = self._sdk(defer_commands=False)
        service = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        state = sdk.state.snapshot()
        targets = [
            {
                "tool_ref": tool_ref,
                "target_pose": {
                    "frame_id": state.end_effectors[side].frame_id,
                    # 接触完成属于真实运动选项，不能用“目标等于当前位姿”的
                    # 空轨迹验证。多末端规划会正确拒绝空轨迹，因此这里给两侧
                    # 末端同一个短插入量，再检查参数是否完整传给 Robot SDK。
                    "position_m": [
                        state.end_effectors[side].position[0] + 0.01,
                        state.end_effectors[side].position[1],
                        state.end_effectors[side].position[2],
                    ],
                    "orientation_xyzw": list(state.end_effectors[side].quaternion_xyzw),
                },
            }
            for tool_ref, side in (
                ("component://tool/left", "left"),
                ("component://tool/right", "right"),
            )
        ]

        request = {
            "robot_id": "r1pro-fake-001",
            "candidate_id": "candidate-contact",
            "targets": targets,
            "coordination": "synchronized",
            "purpose": "insert",
            "object_ref": "object://box-1",
            "target_revision": "target-r1",
            "maximum_speed_mps": 0.08,
            "max_contact_force_n": 25.0,
            "invocation_id": "inv-contact",
        }
        try:
            with patch.object(
                sdk.upper_body,
                "move_end_effectors",
                wraps=sdk.upper_body.move_end_effectors,
            ) as move:
                result = service.invoke("MoveEndEffector", request)
            self.assertEqual(result["status"], "succeeded")
            kwargs = move.call_args.kwargs
            self.assertFalse(kwargs["stop_on_contact"])
            self.assertEqual(
                kwargs["contact_tool_refs"],
                ["component://tool/left", "component://tool/right"],
            )
            self.assertEqual(kwargs["max_contact_force_n"], 25.0)
            self.assertEqual(kwargs["maximum_speed_mps"], 0.08)
            self.assertEqual(kwargs["position_tolerance_rad"], 0.002)
            self.assertTrue(kwargs["preserve_cartesian_path"])
            self.assertNotIn("speed_scale", kwargs)

            seat_targets = []
            for target in targets:
                position = list(target["target_pose"]["position_m"])
                position[2] += 0.01
                seat_targets.append(
                    {
                        **target,
                        "target_pose": {
                            **target["target_pose"],
                            "position_m": position,
                        },
                    }
                )
            seat_request = {
                **request,
                "invocation_id": "inv-seat",
                "purpose": "seat",
                "targets": seat_targets,
            }
            with patch.object(
                sdk.upper_body,
                "move_end_effectors",
                wraps=sdk.upper_body.move_end_effectors,
            ) as seat_move:
                seat_result = service.invoke("MoveEndEffector", seat_request)
            self.assertEqual(seat_result["status"], "succeeded")
            seat_kwargs = seat_move.call_args.kwargs
            self.assertFalse(seat_kwargs["stop_on_contact"])
            self.assertEqual(
                seat_kwargs["contact_tool_refs"],
                ["component://tool/left", "component://tool/right"],
            )
            self.assertEqual(seat_kwargs["max_contact_force_n"], 25.0)
            self.assertEqual(seat_kwargs["position_tolerance_rad"], 0.002)
            self.assertFalse(seat_kwargs["preserve_cartesian_path"])

            unloaded_state = sdk.state.snapshot()
            unloaded_targets = [
                {
                    "tool_ref": tool_ref,
                    "target_pose": {
                        "frame_id": unloaded_state.end_effectors[side].frame_id,
                        "position_m": [
                            unloaded_state.end_effectors[side].position[0],
                            unloaded_state.end_effectors[side].position[1],
                            unloaded_state.end_effectors[side].position[2] - 0.005,
                        ],
                        "orientation_xyzw": list(
                            unloaded_state.end_effectors[side].quaternion_xyzw
                        ),
                    },
                }
                for tool_ref, side in (
                    ("component://tool/left", "left"),
                    ("component://tool/right", "right"),
                )
            ]
            with patch.object(
                sdk.upper_body,
                "move_end_effectors",
                wraps=sdk.upper_body.move_end_effectors,
            ) as unloaded_unseat_move:
                unloaded_unseat = service.invoke(
                    "MoveEndEffector",
                    {
                        **request,
                        "invocation_id": "inv-unloaded-unseat",
                        "purpose": "unseat",
                        "targets": unloaded_targets,
                    },
                )
            self.assertEqual(unloaded_unseat["status"], "succeeded")
            self.assertEqual(
                unloaded_unseat_move.call_args.kwargs["position_tolerance_rad"],
                0.08,
            )

            clearance_request = {
                **request,
                "invocation_id": "inv-secondary-clearance",
                "purpose": "clearance",
                "targets": [targets[1]],
                "required_contact_tools": ["component://tool/left"],
            }
            clearance_state = sdk.state.snapshot()
            left_ref = "component://tool/left"
            coupled_left = clearance_state.tool_states[left_ref].model_copy(
                update={
                    "hook_contact": True,
                    "clamp_contact": True,
                    "hook_force_n": 3.0,
                    "clamp_force_n": 3.0,
                    "hook_support_ratio": 0.8,
                }
            )
            coupled_load = ToolLoadAssessment(
                engaged=True,
                slipping=False,
                overloaded=False,
                sensor_fault=False,
                reasons=(),
                tool_states={left_ref: coupled_left},
            )
            supported_state = sdk.state.snapshot()
            supported_targets = [
                {
                    "tool_ref": tool_ref,
                    "target_pose": {
                        "frame_id": supported_state.end_effectors[side].frame_id,
                        "position_m": [
                            supported_state.end_effectors[side].position[0] + 0.002,
                            supported_state.end_effectors[side].position[1],
                            supported_state.end_effectors[side].position[2],
                        ],
                        "orientation_xyzw": list(
                            supported_state.end_effectors[side].quaternion_xyzw
                        ),
                    },
                }
                for tool_ref, side in (
                    ("component://tool/left", "left"),
                    ("component://tool/right", "right"),
                )
            ]
            # 支撑面短推只移动仍贴着箱体的一只工具；另一只工具已完成撤离。
            # 使用单目标才能覆盖Ability中这条局部执行容差，而不会误伤普通
            # 双工具 disengage。
            supported_targets = supported_targets[:1]
            with patch.object(
                sdk.upper_body,
                "move_end_effector",
                wraps=sdk.upper_body.move_end_effector,
            ) as supported_move:
                supported_result = service.invoke(
                    "MoveEndEffector",
                    {
                        **request,
                        "invocation_id": "inv-supported-slide",
                        "purpose": "disengage",
                        "targets": supported_targets,
                        "required_contact_tools": [],
                        "expected_object_pose": {
                            "frame_id": "world",
                            "position_m": [1.2, 0.8, 0.32],
                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                )
            self.assertEqual(supported_result["status"], "succeeded")
            self.assertEqual(
                supported_move.call_args.kwargs["position_tolerance_rad"],
                0.05,
            )
            with (
                patch(
                    "r1pro_abilities.handlers.manipulator_motion.read_tool_load",
                    return_value=(clearance_state, coupled_load),
                ),
                patch(
                    "r1pro_abilities.handlers.manipulator_motion.scene_collision_environment",
                    wraps=scene_collision_environment,
                ) as clearance_environment,
                patch.object(
                    sdk.upper_body,
                    "move_end_effector",
                    wraps=sdk.upper_body.move_end_effector,
                ) as clearance_move,
                patch.object(
                    sdk.upper_body,
                    "move_end_effectors",
                    wraps=sdk.upper_body.move_end_effectors,
                ) as clearance_multi_move,
            ):
                clearance_result = service.invoke("MoveEndEffector", clearance_request)
            self.assertEqual(clearance_result["status"], "succeeded")
            self.assertFalse(clearance_move.call_args.kwargs["preserve_cartesian_path"])
            self.assertEqual(clearance_move.call_args.args[0], "right")
            clearance_multi_move.assert_not_called()

            # 放置后的空手上抬同样使用 clearance，但 expected_object_pose
            # 明确表明箱体已经落座。此时若另一侧仍接触箱体，必须把该侧
            # 实时末端作为固定端同步规划，不能退化成会带动躯干的单臂动作。
            placement_clearance_state = sdk.state.snapshot()
            placement_clearance_pose = placement_clearance_state.end_effectors["right"]
            placement_clearance_request = {
                **clearance_request,
                "invocation_id": "inv-placement-clearance",
                "expected_object_pose": POSE,
                "targets": [{
                    "tool_ref": "component://tool/right",
                    "target_pose": {
                        "frame_id": placement_clearance_pose.frame_id,
                        "position_m": [
                            placement_clearance_pose.position[0],
                            placement_clearance_pose.position[1],
                            placement_clearance_pose.position[2] + 0.001,
                        ],
                        "orientation_xyzw": list(
                            placement_clearance_pose.quaternion_xyzw
                        ),
                    },
                }],
            }
            with (
                patch(
                    "r1pro_abilities.handlers.manipulator_motion.read_tool_load",
                    return_value=(placement_clearance_state, coupled_load),
                ),
                patch(
                    "r1pro_abilities.handlers.manipulator_motion.scene_collision_environment",
                    wraps=scene_collision_environment,
                ) as placement_clearance_environment,
                patch.object(
                    sdk.upper_body,
                    "move_end_effectors",
                    wraps=sdk.upper_body.move_end_effectors,
                ) as placement_clearance_move,
            ):
                placement_clearance_result = service.invoke(
                    "MoveEndEffector", placement_clearance_request
                )
            self.assertEqual(placement_clearance_result["status"], "succeeded")
            self.assertEqual(
                placement_clearance_move.call_args.kwargs["fixed_end_effectors"],
                {"left"},
            )
            self.assertEqual(
                set(placement_clearance_move.call_args.args[0]), {"left", "right"}
            )
            self.assertTrue(
                placement_clearance_move.call_args.kwargs["preserve_cartesian_path"]
            )
            self.assertEqual(
                placement_clearance_environment.call_args.kwargs[
                    "exclude_source_ids"
                ],
                ("object://box-1",),
            )
            alignment_state = sdk.state.snapshot()
            alignment_pose = alignment_state.end_effectors["right"]
            alignment_target = {
                "tool_ref": "component://tool/right",
                "target_pose": {
                    "frame_id": alignment_pose.frame_id,
                    "position_m": [
                        alignment_pose.position[0] + 0.001,
                        alignment_pose.position[1],
                        alignment_pose.position[2],
                    ],
                    "orientation_xyzw": list(alignment_pose.quaternion_xyzw),
                },
            }
            alignment_request = {
                **clearance_request,
                "invocation_id": "inv-secondary-alignment",
                "purpose": "alignment",
                "targets": [alignment_target],
            }
            with (
                patch(
                    "r1pro_abilities.handlers.manipulator_motion.read_tool_load",
                    return_value=(clearance_state, coupled_load),
                ),
                patch.object(
                    sdk.upper_body,
                    "move_end_effectors",
                    wraps=sdk.upper_body.move_end_effectors,
                ) as alignment_move,
            ):
                alignment_result = service.invoke("MoveEndEffector", alignment_request)
            self.assertEqual(alignment_result["status"], "succeeded")
            self.assertEqual(
                alignment_move.call_args.kwargs["fixed_end_effectors"], {"left"}
            )
            self.assertEqual(set(alignment_move.call_args.args[0]), {"left", "right"})

            insert_state = sdk.state.snapshot()
            insert_pose = insert_state.end_effectors["right"]
            insert_request = {
                **clearance_request,
                "invocation_id": "inv-secondary-insert",
                "purpose": "insert",
                "targets": [
                    {
                        "tool_ref": "component://tool/right",
                        "target_pose": {
                            "frame_id": insert_pose.frame_id,
                            "position_m": [
                                insert_pose.position[0] + 0.001,
                                insert_pose.position[1],
                                insert_pose.position[2],
                            ],
                            "orientation_xyzw": list(insert_pose.quaternion_xyzw),
                        },
                    }
                ],
            }
            with (
                patch(
                    "r1pro_abilities.handlers.manipulator_motion.read_tool_load",
                    return_value=(clearance_state, coupled_load),
                ),
                patch.object(
                    sdk.upper_body,
                    "move_end_effector",
                    wraps=sdk.upper_body.move_end_effector,
                ) as insert_single_move,
                patch.object(
                    sdk.upper_body,
                    "move_end_effectors",
                    wraps=sdk.upper_body.move_end_effectors,
                ) as insert_move,
            ):
                insert_result = service.invoke("MoveEndEffector", insert_request)
            self.assertEqual(insert_result["status"], "succeeded")
            self.assertEqual(set(insert_move.call_args.args[0]), {"left", "right"})
            self.assertEqual(
                insert_move.call_args.kwargs["fixed_end_effectors"], {"left"}
            )
            self.assertTrue(insert_move.call_args.kwargs["preserve_cartesian_path"])
            insert_single_move.assert_not_called()

            # required_contact_tools只监控并放行首侧与目标箱的既有接触；
            # 运动中的第二侧和邻箱仍参与碰撞检查。
            self.assertEqual(
                clearance_environment.call_args.kwargs["exclude_source_ids"], ()
            )
            self.assertEqual(
                clearance_environment.call_args.kwargs[
                    "allowed_contact_end_effectors_by_source"
                ],
                {"object://box-1": ("left",)},
            )

            extract_sdk = self._sdk(defer_commands=False)
            extract_service = R1ProAbilityService(
                AbilityRole.MANIPULATOR_MOTION, extract_sdk
            )
            try:
                extract_state = extract_sdk.state.snapshot()
                extract_pose = extract_state.end_effectors["left"]
                extract_target = {
                    "tool_ref": "component://tool/left",
                    "target_pose": {
                        "frame_id": extract_pose.frame_id,
                        "position_m": [
                            extract_pose.position[0] + 0.05,
                            extract_pose.position[1],
                            extract_pose.position[2],
                        ],
                        "orientation_xyzw": list(extract_pose.quaternion_xyzw),
                    },
                }
                extract_request = {
                    **request,
                    "invocation_id": "inv-constrained-extract",
                    "purpose": "extract",
                    "targets": [extract_target],
                    "required_contact_tools": ["component://tool/left"],
                }
                extract_left = extract_state.tool_states[left_ref].model_copy(
                    update={
                        "hook_contact": True,
                        "clamp_contact": True,
                        "hook_force_n": 3.0,
                        "clamp_force_n": 3.0,
                        "hook_support_ratio": 0.8,
                    }
                )
                extract_coupled_load = ToolLoadAssessment(
                    engaged=coupled_load.engaged,
                    slipping=coupled_load.slipping,
                    overloaded=coupled_load.overloaded,
                    sensor_fault=coupled_load.sensor_fault,
                    reasons=coupled_load.reasons,
                    tool_states={left_ref: extract_left},
                )
                with (
                    patch.object(
                        extract_sdk.upper_body,
                        "move_end_effector",
                        wraps=extract_sdk.upper_body.move_end_effector,
                    ) as extract_move,
                    patch.object(
                        extract_sdk.upper_body,
                        "move_end_effectors",
                        wraps=extract_sdk.upper_body.move_end_effectors,
                    ) as extract_multi_move,
                    patch(
                        "r1pro_abilities.handlers.manipulator_motion.read_tool_load",
                        return_value=(extract_state, extract_coupled_load),
                    ) as extract_load,
                ):
                    extract_result = extract_service.invoke(
                        "MoveEndEffector", extract_request
                    )
                self.assertEqual(extract_result["status"], "succeeded")
                self.assertTrue(
                    extract_move.call_args.kwargs["preserve_cartesian_path"]
                )
                self.assertIsNone(
                    extract_move.call_args.kwargs["position_tolerance_rad"]
                )
                self.assertEqual(extract_move.call_args.args[0], "left")
                extract_multi_move.assert_not_called()
                extract_policy = extract_load.call_args.args[2]
                self.assertEqual(
                    extract_policy.maximum_tangential_speed_m_s,
                    DEFAULT_TOOL_LOAD_POLICY.maximum_tangential_speed_m_s
                    + extract_request["maximum_speed_mps"],
                )
            finally:
                extract_service.close()
                extract_sdk.close()
        finally:
            service.close()
            sdk.close()

    def test_extract_keeps_contact_when_only_relative_motion_is_high(self) -> None:
        """主动外拉的切向运动保留为原始诊断，不能单独冒充脱手。"""

        sdk = self._sdk()
        try:
            state = sdk.state.snapshot()
            left_ref = "component://tool/left"
            left = state.tool_states[left_ref].model_copy(
                update={
                    "hook_contact": True,
                    "clamp_contact": True,
                    "hook_force_n": 5.0,
                    "clamp_force_n": 4.0,
                    "hook_support_ratio": 0.9,
                    "hook_tangential_speed_m_s": 0.054,
                    "sensor_fault": False,
                }
            )
            load = ToolLoadAssessment(
                engaged=True,
                slipping=True,
                overloaded=False,
                sensor_fault=False,
                reasons=(f"{left_ref}:relative_motion_high",),
                tool_states={left_ref: left},
            )
            record = type(
                "ExtractRecord",
                (),
                {"request": {"purpose": "extract"}},
            )()

            maintained, reasons = ManipulatorMotionHandler._required_contact_state(
                record, load
            )
            self.assertTrue(maintained)
            self.assertNotIn(f"{left_ref}:relative_motion_high", reasons)
        finally:
            sdk.close()

    def test_secondary_engagement_stops_when_primary_contact_is_lost(self) -> None:
        """第二侧接近期间首侧持续脱离时，Ability必须停止动作并进入hold。"""

        sdk = self._sdk()
        service = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        state = sdk.state.snapshot()
        target_pose = {
            "frame_id": state.end_effectors["right"].frame_id,
            "position_m": [
                state.end_effectors["right"].position[0] + 0.01,
                state.end_effectors["right"].position[1],
                state.end_effectors["right"].position[2],
            ],
            "orientation_xyzw": list(state.end_effectors["right"].quaternion_xyzw),
        }
        unsafe = ToolLoadAssessment(
            engaged=False,
            slipping=False,
            overloaded=False,
            sensor_fault=False,
            reasons=("component://tool/left:hook_contact_missing",),
            tool_states=dict(state.tool_states),
        )
        request = {
            "robot_id": "r1pro-fake-001",
            "invocation_id": "secondary-primary-contact",
            "candidate_id": "candidate-after-extraction",
            "targets": [
                {
                    "tool_ref": "component://tool/right",
                    "target_pose": target_pose,
                }
            ],
            "coordination": "synchronized",
            "purpose": "pregrasp",
            "object_ref": "object://box-1",
            "target_revision": "target-r2",
            "maximum_speed_mps": 0.05,
            "required_contact_tools": ["component://tool/left"],
        }
        try:
            with (
                patch(
                    "r1pro_abilities.handlers.manipulator_motion.read_tool_load",
                    return_value=(state, unsafe),
                ),
                patch.object(
                    sdk.upper_body,
                    "move_end_effectors",
                    wraps=sdk.upper_body.move_end_effectors,
                ) as move,
            ):
                started = service.invoke("MoveEndEffector", request)
                self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)
                self.assertTrue(move.call_args.kwargs["preserve_cartesian_path"])
                self.assertEqual(move.call_args.kwargs["fixed_end_effectors"], {"left"})
                self.assertEqual(move.call_args.kwargs["position_tolerance_rad"], 0.002)
                time.sleep(0.14)
                stopped = service.invoke(
                    "GetExecution", {"invocation_id": request["invocation_id"]}
                )
            # stop 是安全动作，Ability Task 本身仍应以“接触丢失”失败收敛，
            # 上层 Skill 才能进入恢复，而不是把安全停止误报成业务成功。
            self.assertEqual(stopped["status"], ExecutionStatus.FAILED.value)
            self.assertTrue(stopped["result"]["stop_evidence"]["holding"])
            self.assertTrue(stopped["result"]["hold_confirmed"])
            self.assertEqual(
                stopped["observations"][-1]["kind"],
                "manipulation.required_tool_contact",
            )
            self.assertFalse(stopped["observations"][-1]["value"]["contact_maintained"])
        finally:
            service.close()
            sdk.close()

    def test_secondary_engagement_keeps_coupled_contact_before_lift(self) -> None:
        """箱体仍由托盘支撑时，首侧接触无需提前冒充承载受力。"""

        sdk = self._sdk()
        service = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        state = sdk.state.snapshot()
        left_ref = "component://tool/left"
        left = state.tool_states[left_ref].model_copy(
            update={
                "hook_contact": True,
                "clamp_contact": True,
                "hook_force_n": 0.27,
                "clamp_force_n": 1.07,
                "hook_support_ratio": 0.00007,
                "sensor_fault": False,
            }
        )
        coupled = ToolLoadAssessment(
            engaged=False,
            slipping=False,
            overloaded=False,
            sensor_fault=False,
            reasons=(
                f"{left_ref}:hook_force_low",
                f"{left_ref}:clamp_force_low",
                f"{left_ref}:vertical_support_low",
            ),
            tool_states={left_ref: left},
        )
        target_pose = {
            "frame_id": state.end_effectors["left"].frame_id,
            "position_m": [
                state.end_effectors["left"].position[0] + 0.01,
                state.end_effectors["left"].position[1],
                state.end_effectors["left"].position[2],
            ],
            "orientation_xyzw": list(state.end_effectors["left"].quaternion_xyzw),
        }
        request = {
            "robot_id": "r1pro-fake-001",
            "invocation_id": "secondary-coupled-contact",
            "candidate_id": "candidate-after-extraction",
            "targets": [{"tool_ref": left_ref, "target_pose": target_pose}],
            "coordination": "synchronized",
            "purpose": "seat",
            "object_ref": "object://box-1",
            "target_revision": "target-r2",
            "maximum_speed_mps": 0.05,
            "required_contact_tools": [left_ref],
        }
        try:
            with patch(
                "r1pro_abilities.handlers.manipulator_motion.read_tool_load",
                return_value=(state, coupled),
            ):
                started = service.invoke("MoveEndEffector", request)
                self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)
                time.sleep(0.14)
                refreshed = service.invoke(
                    "GetExecution", {"invocation_id": request["invocation_id"]}
                )
            self.assertEqual(refreshed["status"], ExecutionStatus.RUNNING.value)
            value = refreshed["observations"][-1]["value"]
            self.assertTrue(value["contact_maintained"])
            self.assertNotIn(f"{left_ref}:hook_force_low", value["reasons"])
            self.assertNotIn(f"{left_ref}:clamp_force_low", value["reasons"])
            self.assertNotIn(f"{left_ref}:vertical_support_low", value["reasons"])
        finally:
            service.close()
            sdk.close()

    def test_insert_does_not_stop_on_side_contact_before_seat(self) -> None:
        """水平insert碰到凹槽侧壁时仍要走到终点，不能冒充下钩承载。"""

        sdk = self._sdk(defer_commands=True)
        service = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        state = sdk.state.snapshot()
        left_ref = "component://tool/left"
        current = state.end_effectors["left"]
        request = {
            "robot_id": "r1pro-fake-001",
            "invocation_id": "insert-contact-finish",
            "candidate_id": "candidate-contact",
            "targets": [
                {
                    "tool_ref": left_ref,
                    "target_pose": {
                        "frame_id": current.frame_id,
                        "position_m": [
                            current.position[0] + 0.02,
                            current.position[1],
                            current.position[2],
                        ],
                        "orientation_xyzw": list(current.quaternion_xyzw),
                    },
                }
            ],
            "coordination": "synchronized",
            "purpose": "insert",
            "object_ref": "object://box-1",
            "target_revision": "target-r1",
            "maximum_speed_mps": 0.02,
            "max_contact_force_n": 60.0,
        }
        try:
            started = service.invoke("MoveEndEffector", request)
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)
            command_id = started["command_id"]
            command = sdk.backend._commands[command_id]
            sdk.backend._commands[command_id] = command.model_copy(
                update={"progress": 0.999}
            )
            original = state.tool_states[left_ref]
            contacted = state.model_copy(
                update={
                    "end_effectors": {
                        **state.end_effectors,
                        "left": current.model_copy(
                            update={
                                "position": (
                                    current.position[0] + 0.019,
                                    current.position[1],
                                    current.position[2],
                                )
                            }
                        ),
                    },
                    "tool_states": {
                        **state.tool_states,
                        left_ref: original.model_copy(
                            update={
                                "hook_contact": True,
                                "hook_force_n": 0.4,
                                "hook_support_ratio": 0.0,
                            }
                        ),
                    },
                }
            )
            held = contacted.model_copy(update={"in_hold": True})
            with patch.object(sdk.state, "snapshot", side_effect=[contacted, held]):
                inserted = service.invoke(
                    "GetExecution", {"invocation_id": request["invocation_id"]}
                )

            self.assertEqual(inserted["status"], ExecutionStatus.RUNNING.value)
            self.assertEqual(
                sdk.commands.get(command_id).status,
                CommandState.RUNNING,
            )
        finally:
            service.close()
            sdk.close()

    def test_insert_does_not_request_stop_near_command_completion(self) -> None:
        """insert接近终点的普通接触不应触发stop，命令自行完成。"""

        sdk = self._sdk(defer_commands=True)
        service = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        state = sdk.state.snapshot()
        left_ref = "component://tool/left"
        current = state.end_effectors["left"]
        request = {
            "robot_id": "r1pro-fake-001",
            "invocation_id": "insert-completion-race",
            "candidate_id": "candidate-contact",
            "targets": [
                {
                    "tool_ref": left_ref,
                    "target_pose": {
                        "frame_id": current.frame_id,
                        "position_m": [
                            current.position[0] + 0.02,
                            current.position[1],
                            current.position[2],
                        ],
                        "orientation_xyzw": list(current.quaternion_xyzw),
                    },
                }
            ],
            "coordination": "synchronized",
            "purpose": "insert",
            "object_ref": "object://box-1",
            "target_revision": "target-r1",
            "maximum_speed_mps": 0.02,
            "max_contact_force_n": 60.0,
        }
        try:
            started = service.invoke("MoveEndEffector", request)
            command_id = started["command_id"]
            command = sdk.backend._commands[command_id]
            sdk.backend._commands[command_id] = command.model_copy(
                update={"progress": 0.999}
            )
            original = state.tool_states[left_ref]
            contacted = state.model_copy(
                update={
                    "end_effectors": {
                        **state.end_effectors,
                        "left": current.model_copy(
                            update={
                                "position": (
                                    current.position[0] + 0.019,
                                    current.position[1],
                                    current.position[2],
                                )
                            }
                        ),
                    },
                    "tool_states": {
                        **state.tool_states,
                        left_ref: original.model_copy(
                            update={
                                "hook_contact": True,
                                "hook_force_n": 0.4,
                                "hook_support_ratio": 0.0,
                            }
                        ),
                    },
                }
            )
            held = contacted.model_copy(update={"in_hold": True})
            completed = command.model_copy(
                update={"status": CommandState.SUCCEEDED, "progress": 1.0}
            )
            with patch.object(
                sdk.safety, "stop_and_hold", return_value=completed
            ) as stop_and_hold:
                inserted = service.invoke(
                    "GetExecution", {"invocation_id": request["invocation_id"]}
                )

            self.assertEqual(inserted["status"], ExecutionStatus.RUNNING.value)
            stop_and_hold.assert_not_called()
        finally:
            service.close()
            sdk.close()

    def test_seat_finishes_on_groove_ceiling_before_geometric_endpoint(self) -> None:
        """下钩脚贴住凹槽内上表面后停止，再由close验证持续夹持。"""

        sdk = self._sdk(defer_commands=True)
        service = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        state = sdk.state.snapshot()
        left_ref = "component://tool/left"
        current = state.end_effectors["left"]
        request = {
            "robot_id": "r1pro-fake-001",
            "invocation_id": "seat-contact-stop",
            "candidate_id": "candidate-contact",
            "targets": [
                {
                    "tool_ref": left_ref,
                    "target_pose": {
                        "frame_id": current.frame_id,
                        "position_m": [
                            current.position[0],
                            current.position[1],
                            current.position[2] + 0.02,
                        ],
                        "orientation_xyzw": list(current.quaternion_xyzw),
                    },
                }
            ],
            "coordination": "synchronized",
            "purpose": "seat",
            "object_ref": "object://box-1",
            "target_revision": "target-r1",
            "maximum_speed_mps": 0.02,
            "max_contact_force_n": 60.0,
        }
        try:
            started = service.invoke("MoveEndEffector", request)
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)

            original = state.tool_states[left_ref]
            glancing = state.model_copy(
                update={
                    "tool_states": {
                        **state.tool_states,
                        left_ref: original.model_copy(
                            update={
                                "hook_contact": True,
                                "hook_force_n": 0.1,
                                "hook_support_ratio": 0.01,
                            }
                        ),
                    }
                }
            )
            with patch.object(sdk.state, "snapshot", return_value=glancing):
                moving = service.invoke(
                    "GetExecution", {"invocation_id": request["invocation_id"]}
                )
            self.assertEqual(moving["status"], ExecutionStatus.RUNNING.value)

            supported = state.model_copy(
                update={
                    "end_effectors": {
                        **state.end_effectors,
                        "left": current.model_copy(
                            update={
                                "position": (
                                    current.position[0],
                                    current.position[1],
                                    current.position[2] + 0.0015,
                                )
                            }
                        ),
                    },
                    "tool_states": {
                        **state.tool_states,
                        left_ref: original.model_copy(
                            update={
                                "hook_contact": True,
                                "hook_force_n": 0.2,
                                "hook_support_ratio": 0.9,
                            }
                        ),
                    },
                }
            )
            held = supported.model_copy(update={"in_hold": True})
            with patch.object(sdk.state, "snapshot", side_effect=[supported, held]):
                seated = service.invoke(
                    "GetExecution", {"invocation_id": request["invocation_id"]}
                )

            self.assertEqual(seated["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertTrue(seated["result"]["contact_seated"])
            self.assertTrue(seated["result"]["stop_evidence"]["holding"])
            self.assertEqual(
                seated["observations"][-1]["kind"],
                "manipulation.seat_contact",
            )
            self.assertEqual(
                sdk.commands.get(started["command_id"]).status,
                CommandState.STOPPED,
            )
        finally:
            service.close()
            sdk.close()

    def test_manipulator_planning_failure_is_registered_before_command(self) -> None:
        sdk = self._sdk(defer_commands=False)
        service = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        request = {
            "robot_id": "r1pro-fake-001",
            "candidate_id": "candidate-unreachable",
            "targets": [
                {"tool_ref": "component://tool/left", "target_pose": POSE},
                {"tool_ref": "component://tool/right", "target_pose": POSE},
            ],
            "coordination": "synchronized",
            "purpose": "pregrasp",
            "object_ref": "object://box-1",
            "target_revision": "target-r1",
            "maximum_speed_mps": 0.15,
            "invocation_id": "inv-planning-failed",
        }
        try:
            with patch.object(
                sdk.upper_body,
                "move_end_effectors",
                side_effect=PlanningError("多末端 IK 不可达"),
            ):
                result = service.invoke("MoveEndEffector", request)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"]["code"], "PLANNING_FAILED")
            persisted = service.invoke(
                "GetExecution",
                {"invocation_id": "inv-planning-failed"},
            )
            self.assertEqual(persisted["status"], "failed")
        finally:
            service.close()
            sdk.close()

    def test_navigation_resolves_slot_to_live_support_work_pose(self) -> None:
        sdk = self._sdk(defer_commands=False)
        sdk.deployment.robot.sdk.options.update(
            {
                "base_footprint_radius_m": 0.42,
                "navigation_resolution_m": 0.05,
                "manipulation_work_distance_m": 0.55,
            }
        )
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        snapshot = SceneSnapshot(
            scene_key="depalletizing",
            instance_id="scene-1",
            generation=sdk.state.snapshot().generation,
            coordinate_frame="world",
            objects=[
                SceneObject(
                    source_id="pallet-b",
                    category="pallet",
                    name="pallet-b",
                    pose=Pose(
                        frame_id="world",
                        position=(1.5, 1.5, 0.075),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(1.2, 1.0, 0.15),
                )
            ],
            regions=[
                SceneRegion(
                    source_id="pallet-b-slot-r1-c1",
                    name="slot-r1-c1",
                    pose=Pose(
                        frame_id="world",
                        position=(1.19, 1.29, 0.16),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(0.56, 0.36, 0.02),
                    properties={"support_surface_ref": "pallet-b"},
                )
            ],
            observed_at=datetime.now().astimezone(),
        )
        request = {
            "robot_id": "r1pro-fake-001",
            "target": {
                "target_ref": "pallet-b-slot-r1-c1",
                "pose": {
                    "frame_id": "world",
                    "position_m": [1.19, 1.29, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "constraints": {},
            },
            "navigation_purpose": "carry_to_place",
            "maximum_speed_mps": 0.2,
            "minimum_clearance_m": 0.05,
            "invocation_id": "inv-route-live-work-pose",
        }
        try:
            with patch.object(sdk.state, "scene_snapshot", return_value=snapshot):
                result = service.invoke("PlanRoute", request)
            self.assertEqual(result["status"], "succeeded")
            resolved = result["result"]["resolved_target"]["pose"]
            self.assertAlmostEqual(resolved["position_m"][0], 1.19, places=2)
            self.assertLess(resolved["position_m"][1], 0.6)
            self.assertGreater(resolved["orientation_xyzw"][2], 0.7)
        finally:
            service.close()
            sdk.close()

    def test_navigation_carrying_prefers_target_aligned_wide_face(self) -> None:
        sdk = self._sdk(defer_commands=False)
        sdk.deployment.robot.sdk.options.update(
            {
                "base_footprint_radius_m": 0.44,
                "navigation_resolution_m": 0.05,
                "manipulation_work_distance_m": 0.48,
            }
        )
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        snapshot = SceneSnapshot(
            scene_key="depalletizing",
            instance_id="scene-1",
            generation=sdk.state.snapshot().generation,
            coordinate_frame="world",
            objects=[
                SceneObject(
                    source_id="pallet-b",
                    category="pallet",
                    name="pallet-b",
                    pose=Pose(
                        frame_id="world",
                        position=(1.5, 1.5, 0.075),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(1.2, 1.0, 0.15),
                )
            ],
            regions=[
                SceneRegion(
                    source_id="pallet-b-slot-r1-c2",
                    name="slot-r1-c2",
                    pose=Pose(
                        frame_id="world",
                        position=(1.81, 1.29, 0.16),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(0.56, 0.36, 0.02),
                    properties={"support_surface_ref": "pallet-b"},
                )
            ],
            observed_at=datetime.now().astimezone(),
        )
        state = sdk.state.snapshot().model_copy(
            update={
                "base_pose": Pose(
                    frame_id="world",
                    position=(0.31, 0.525, 0.01),
                    quaternion_xyzw=(0.0, 0.0, 0.7071068, 0.7071068),
                )
            }
        )
        request = {
            "target": {
                "target_ref": "pallet-b-slot-r1-c2",
                "pose": {
                    "frame_id": "world",
                    "position_m": [1.81, 1.29, 0.16],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "navigation_purpose": "carry_to_place",
            "minimum_clearance_m": 0.05,
            "carrying_object": {
                "object_ref": "tote-large-l3-r1-c2",
                "object_pose": {
                    "frame_id": "world",
                    "position_m": [0.31, 1.07, 1.08],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "object_size_m": [0.6, 0.4, 0.34],
            },
        }
        try:
            with (
                patch.object(sdk.state, "scene_snapshot", return_value=snapshot),
                patch.object(sdk.state, "snapshot", return_value=state),
            ):
                goals = service.handler._route_goal_candidates(request)
            # 东侧工位更接近且工作距离误差更小，但会把箱体旋转
            # 90 度。携物放置必须选南侧宽面工位，保持箱体与槽位同向。
            self.assertAlmostEqual(goals[0].position[0], 1.81, places=2)
            self.assertLess(goals[0].position[1], 0.6)
            self.assertGreater(goals[0].quaternion_xyzw[2], 0.7)
        finally:
            service.close()
            sdk.close()

    def test_navigation_grasp_keeps_wide_face_when_work_distance_is_closer(self) -> None:
        """调近工作距离后仍应从宽面接近，而不是追求窄面数值误差。"""

        sdk = self._sdk(defer_commands=False)
        sdk.deployment.robot.sdk.options.update(
            {
                "base_footprint_radius_m": 0.44,
                "navigation_resolution_m": 0.05,
                "manipulation_work_distance_m": 0.48,
            }
        )
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        snapshot = SceneSnapshot(
            scene_key="depalletizing",
            instance_id="scene-1",
            generation=sdk.state.snapshot().generation,
            coordinate_frame="world",
            objects=[
                SceneObject(
                    source_id="pallet-a",
                    category="pallet",
                    name="pallet-a",
                    pose=Pose(
                        frame_id="world",
                        position=(0.0, 1.5, 0.075),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(1.2, 1.0, 0.15),
                ),
                SceneObject(
                    source_id="tote-large-l3-r1-c1",
                    category="tote",
                    name="tote-large-l3-r1-c1",
                    pose=Pose(
                        frame_id="world",
                        position=(-0.31, 1.29, 1.0),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(0.6, 0.4, 0.34),
                ),
            ],
            regions=[],
            observed_at=datetime.now().astimezone(),
        )
        state = sdk.state.snapshot().model_copy(
            update={
                "base_pose": Pose(
                    frame_id="world",
                    # 即使Robot当前更靠近北侧，也保持R1 Pro双臂已验证的
                    # 标准负Y宽面工位，不能因少走一段路翻转抓取构型。
                    position=(0.0, 3.0, 0.01),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                )
            }
        )
        request = {
            "target": {
                "target_ref": "tote-large-l3-r1-c1",
                "pose": {
                    "frame_id": "world",
                    "position_m": [-0.31, 1.29, 1.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "navigation_purpose": "approach_grasp",
            "minimum_clearance_m": 0.05,
        }
        try:
            with (
                patch.object(sdk.state, "scene_snapshot", return_value=snapshot),
                patch.object(sdk.state, "snapshot", return_value=state),
            ):
                goals = service.handler._route_goal_candidates(request)
            self.assertAlmostEqual(goals[0].position[0], -0.31, places=2)
            self.assertLess(goals[0].position[1], 0.6)
            self.assertGreater(goals[0].quaternion_xyzw[2], 0.7)
        finally:
            service.close()
            sdk.close()

    def test_navigation_planning_failure_is_registered_before_command(self) -> None:
        sdk = self._sdk(defer_commands=False)
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        request = {
            "robot_id": "r1pro-fake-001",
            "target": TARGET,
            "navigation_purpose": "approach_grasp",
            "maximum_speed_mps": 0.2,
            "minimum_clearance_m": 0.05,
            "invocation_id": "inv-route-planning-failed",
        }
        try:
            with patch.object(
                sdk.base,
                "plan_route",
                side_effect=PlanningError("终点位于障碍物中"),
            ):
                result = service.invoke("PlanRoute", request)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"]["code"], "PLANNING_FAILED")
            persisted = service.invoke(
                "GetExecution",
                {"invocation_id": "inv-route-planning-failed"},
            )
            self.assertEqual(persisted["status"], "failed")
        finally:
            service.close()
            sdk.close()

    def test_navigation_planning_passes_live_carried_object_to_sdk(self) -> None:
        sdk = self._sdk(defer_commands=False)
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        request = {
            "robot_id": "r1pro-fake-001",
            "target": TARGET,
            "navigation_purpose": "carry_to_place",
            "maximum_speed_mps": 0.2,
            "minimum_clearance_m": 0.05,
            "carrying_object": {
                "object_ref": "object://pallet-a/box-17",
                "object_pose": POSE,
                "object_size_m": [0.6, 0.4, 0.34],
            },
            "invocation_id": "inv-route-carrying",
        }
        try:
            with patch.object(
                sdk.base,
                "plan_route",
                wraps=sdk.base.plan_route,
            ) as plan_route:
                result = service.invoke("PlanRoute", request)

            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(
                result["result"]["resolved_target"]["target_ref"],
                TARGET["target_ref"],
            )
            self.assertEqual(
                result["result"]["resolved_target"]["pose"]["position_m"],
                TARGET["pose"]["position_m"],
            )
            self.assertEqual(
                plan_route.call_args.kwargs["carrying_object_ref"],
                "object://pallet-a/box-17",
            )
            self.assertEqual(
                plan_route.call_args.kwargs["carrying_object_pose"].position,
                (1.0, 2.0, 0.0),
            )
            self.assertEqual(
                plan_route.call_args.kwargs["carrying_object_extent_m"],
                (0.6, 0.4, 0.34),
            )
        finally:
            service.close()
            sdk.close()

    def test_formal_grasp_actions_return_skill_observations(self) -> None:
        sdk = self._sdk(defer_commands=False)
        models = ModelProfileRegistry.load(ROOT / "configs/models.example.json")
        perception = R1ProAbilityService(AbilityRole.OBJECT_PERCEPTION, sdk, models)
        planning = R1ProAbilityService(AbilityRole.GRASP_PLANNING, sdk, models)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        gripper = R1ProAbilityService(AbilityRole.END_EFFECTOR, sdk)
        try:
            located = perception.invoke(
                "LocateObject",
                {
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "minimum_confidence": 0.65,
                },
            )
            target_observation = located["observations"][0]
            self.assertEqual(target_observation["kind"], "target_pose")
            target_value = target_observation["value"]

            candidates = planning.invoke(
                "GenerateCandidates",
                {
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "target_pose": target_value["pose"],
                    "object_extent_m": target_value["extent_m"],
                    "target_revision": target_observation["revision"],
                    "preferred_strategy": "auto",
                    "maximum_candidates": 4,
                },
            )
            candidate = candidates["observations"][0]["value"]["candidates"][0]

            opened = gripper.invoke(
                "SetOpening",
                {
                    "robot_id": "r1pro-fake-001",
                    "tools": candidate["opening_setpoints"],
                },
            )
            self.assertEqual(opened["status"], "succeeded")
            self.assertTrue(opened["result"]["verified"])
            self.assertEqual(
                opened["observations"][0]["kind"], "manipulation.tool_opening"
            )

            moved = motion.invoke(
                "MoveEndEffector",
                {
                    "robot_id": "r1pro-fake-001",
                    "candidate_id": candidate["candidate_id"],
                    "targets": candidate["approach_poses"],
                    "coordination": "synchronized",
                    "purpose": "pregrasp",
                    "object_ref": "object://box-1",
                    "target_revision": target_observation["revision"],
                    "maximum_speed_mps": 0.15,
                },
            )
            self.assertEqual(moved["status"], "succeeded")
            self.assertEqual(moved["result"]["command_completed"], True)
            self.assertEqual(moved["observations"], [])

            sdk.backend.configure_grasp_fixture(
                "left", "object://box-1", contact_opening_m=0.06, contact_force_n=18.0
            )
            sdk.backend.configure_grasp_fixture(
                "right", "object://box-1", contact_opening_m=0.06, contact_force_n=18.0
            )
            grasped = gripper.invoke(
                "CloseUntilContact",
                {
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "tools": candidate["clamp_setpoints"],
                    "candidate_id": candidate["candidate_id"],
                    "grasp_pose": target_value["pose"],
                },
            )
            self.assertEqual(
                grasped["observations"][0]["kind"], "manipulation.tool_closed"
            )

            lifted = motion.invoke(
                "LiftHeldObject",
                {
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "tools": [
                        "component://tool/left",
                        "component://tool/right",
                    ],
                    "candidate_id": candidate["candidate_id"],
                    "distance_m": 0.1,
                    "maximum_speed_mps": 0.1,
                },
            )
            self.assertEqual(lifted["operation"], "LiftHeldObject")
            self.assertEqual(lifted["observations"][0]["kind"], "lift_progress")
            self.assertEqual(lifted["observations"][0]["value"]["lift_height_m"], 0.1)
        finally:
            perception.close()
            planning.close()
            motion.close()
            gripper.close()
            sdk.close()

    def test_formal_place_actions_return_independent_evidence(self) -> None:
        sdk = self._sdk(defer_commands=False)
        models = ModelProfileRegistry.load(ROOT / "configs/models.example.json")
        state = R1ProAbilityService(AbilityRole.ROBOT_STATE, sdk)
        perception = R1ProAbilityService(AbilityRole.OBJECT_PERCEPTION, sdk, models)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        gripper = R1ProAbilityService(AbilityRole.END_EFFECTOR, sdk)
        try:
            self._engage_bilateral_tools(sdk)
            snapshot = state.invoke("GetRobotState", {"robot_id": "r1pro-fake-001"})
            snapshot_value = snapshot["observations"][0]["value"]
            self.assertEqual(snapshot["observations"][0]["kind"], "robot.state")
            self.assertIn("base_pose", snapshot_value)
            self.assertIn("end_effectors", snapshot_value)
            self.assertIn("tool_states", snapshot_value)
            self.assertNotIn("stable_load", snapshot_value)
            self.assertNotIn("tool_loads", snapshot_value)

            verified_load = state.invoke(
                "VerifyToolLoad",
                {
                    "robot_id": "r1pro-fake-001",
                    "tool_refs": [
                        "component://tool/left",
                        "component://tool/right",
                    ],
                },
            )
            self.assertEqual(
                verified_load["observations"][0]["kind"], "robot.tool_load"
            )
            load_value = verified_load["observations"][0]["value"]
            self.assertTrue(load_value["condition_satisfied"])
            self.assertGreaterEqual(load_value["observed_duration_ms"], 120)
            self.assertEqual(
                {item["tool_ref"] for item in load_value["tools"]},
                {"component://tool/left", "component://tool/right"},
            )
            self.assertTrue(all(item["hook_contact"] for item in load_value["tools"]))
            self.assertNotIn("object_ref", load_value)
            self.assertNotIn("object_pose", load_value)

            observed = perception.invoke(
                "ObservePlacementTarget",
                {
                    "robot_id": "r1pro-fake-001",
                    "target_ref": "slot://pallet-b/cell-1",
                    "object_ref": "object://box-1",
                },
            )
            slot = observed["observations"][0]["value"]
            for waypoint_name in ("preplace", "release"):
                approach = motion.invoke(
                    "MoveEndEffector",
                    {
                        "invocation_id": f"place-{waypoint_name}",
                        "robot_id": "r1pro-fake-001",
                        "object_ref": "object://box-1",
                        "target_ref": slot["target_ref"],
                        "target_revision": slot["revision"],
                        "targets": [
                            {
                                "tool_ref": "component://tool/left",
                                "target_pose": slot["placement_pose"],
                            },
                            {
                                "tool_ref": "component://tool/right",
                                "target_pose": slot["placement_pose"],
                            },
                        ],
                        "coordination": "synchronized",
                        "purpose": "place",
                        "expected_object_pose": slot["placement_pose"],
                        "maximum_speed_mps": 0.08,
                        "max_contact_force_n": 20.0,
                    },
                )
                self.assertEqual(approach["status"], "succeeded")

            released = gripper.invoke(
                "Release",
                {
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "tools": [
                        {
                            "tool_ref": "component://tool/left",
                            "target_position_m": 0.035,
                            "maximum_force_n": 60.0,
                            "hold": False,
                        },
                        {
                            "tool_ref": "component://tool/right",
                            "target_position_m": 0.035,
                            "maximum_force_n": 60.0,
                            "hold": False,
                        },
                    ],
                    "target_ref": slot["target_ref"],
                    "target_revision": slot["revision"],
                },
            )
            self.assertEqual(
                released["observations"][0]["kind"], "manipulation.object_released"
            )

            retreat_state = sdk.state.snapshot()
            retreat_targets = []
            for tool_ref, side in (
                ("component://tool/left", "left"),
                ("component://tool/right", "right"),
            ):
                end_effector = retreat_state.end_effectors[side]
                retreat_targets.append(
                    {
                        "tool_ref": tool_ref,
                        "target_pose": {
                            "frame_id": end_effector.frame_id,
                            "position_m": [
                                end_effector.position[0],
                                end_effector.position[1],
                                end_effector.position[2] + 0.01,
                            ],
                            "orientation_xyzw": list(end_effector.quaternion_xyzw),
                        },
                    }
                )
            retreat_patch = patch.object(
                sdk.upper_body,
                "move_end_effectors",
                wraps=sdk.upper_body.move_end_effectors,
            )
            retreat_move = retreat_patch.start()
            self.addCleanup(retreat_patch.stop)
            retreat_environment_patch = patch(
                "r1pro_abilities.handlers.manipulator_motion.scene_collision_environment",
                wraps=scene_collision_environment,
            )
            retreat_environment = retreat_environment_patch.start()
            self.addCleanup(retreat_environment_patch.stop)
            retreated = motion.invoke(
                "MoveEndEffector",
                {
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "target_ref": slot["target_ref"],
                    "target_revision": slot["revision"],
                    "targets": retreat_targets,
                    "coordination": "synchronized",
                    "purpose": "retreat",
                    "maximum_speed_mps": 0.08,
                },
            )
            self.assertEqual(retreated["status"], "succeeded")
            self.assertTrue(retreat_move.call_args.kwargs["preserve_cartesian_path"])
            self.assertEqual(
                retreat_move.call_args.kwargs["position_tolerance_rad"], 0.012
            )
            self.assertEqual(
                retreat_environment.call_args.kwargs["exclude_source_ids"], ()
            )

            verified = perception.invoke(
                "VerifyPlacement",
                {
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "target_ref": slot["target_ref"],
                    "target_revision": slot["revision"],
                    "stability_duration_ms": 800,
                },
            )
            evidence = verified["observations"][0]
            self.assertEqual(evidence["kind"], "placement.object_stability")
            self.assertTrue(evidence["value"]["independent_verification"])
            self.assertEqual(evidence["value"]["observed_displacement_m"], 0.0)
            self.assertEqual(evidence["value"]["observed_duration_ms"], 800)
            self.assertTrue(evidence["value"]["support_contact"])
        finally:
            state.close()
            perception.close()
            motion.close()
            gripper.close()
            sdk.close()

    def test_lift_stops_and_holds_when_bilateral_load_is_lost(self) -> None:
        sdk = self._sdk(defer_commands=True)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        try:
            self._engage_bilateral_tools(sdk)
            started = motion.invoke(
                "LiftHeldObject",
                {
                    "invocation_id": "lift-loses-load",
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "tools": [
                        "component://tool/left",
                        "component://tool/right",
                    ],
                    "candidate_id": "candidate-bilateral",
                    "distance_m": 0.08,
                    "maximum_speed_mps": 0.05,
                },
            )
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)

            # 模拟真实抬升中一侧钩爪脱离。Ability允许一个传感刷新周期，
            # 但异常持续存在后必须停止并取得hold，不能等轨迹自然结束。
            release_id = "test-release-right-during-lift"
            sdk.end_effector.release("right", command_id=release_id)
            sdk.backend.complete_command(release_id)
            with patch(
                "r1pro_abilities.handlers.manipulator_motion.load_failure_requires_stop",
                return_value=True,
            ):
                refreshed = motion.invoke(
                    "GetExecution", {"invocation_id": "lift-loses-load"}
                )

            self.assertEqual(refreshed["status"], ExecutionStatus.FAILED.value)
            self.assertEqual(refreshed["error"]["code"], "LOAD_UNSTABLE_DURING_LIFT")
            self.assertTrue(refreshed["result"]["hold_confirmed"])
            self.assertTrue(sdk.state.snapshot().in_hold)
        finally:
            motion.close()
            sdk.close()

    def test_place_motion_stops_and_holds_when_bilateral_load_is_lost(self) -> None:
        sdk = self._sdk(defer_commands=True)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        try:
            self._engage_bilateral_tools(sdk)
            state = sdk.state.snapshot()
            targets = []
            for side in ("left", "right"):
                current = state.end_effectors[side]
                targets.append(
                    {
                        "tool_ref": f"component://tool/{side}",
                        "target_pose": {
                            "frame_id": current.frame_id,
                            "position_m": [
                                current.position[0],
                                current.position[1],
                                current.position[2] + 0.01,
                            ],
                            "orientation_xyzw": list(current.quaternion_xyzw),
                            "observed_at": datetime.now().astimezone().isoformat(),
                            "revision": "place-current-state",
                        },
                    }
                )
            started = motion.invoke(
                "MoveEndEffector",
                {
                    "invocation_id": "place-loses-load",
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "target_ref": "slot://pallet-b/cell-1",
                    "target_revision": "slot-r1",
                    "targets": targets,
                    "coordination": "synchronized",
                    "purpose": "place",
                    "expected_object_pose": POSE,
                    "maximum_speed_mps": 0.04,
                    "max_contact_force_n": 60.0,
                },
            )
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)

            release_id = "test-release-left-during-place"
            sdk.end_effector.release("left", command_id=release_id)
            sdk.backend.complete_command(release_id)
            with patch(
                "r1pro_abilities.handlers.manipulator_motion.load_failure_requires_stop",
                return_value=True,
            ):
                refreshed = motion.invoke(
                    "GetExecution", {"invocation_id": "place-loses-load"}
                )

            self.assertEqual(refreshed["status"], ExecutionStatus.FAILED.value)
            self.assertEqual(
                refreshed["error"]["code"], "LOAD_UNSTABLE_DURING_PLACEMENT"
            )
            self.assertTrue(refreshed["result"]["hold_confirmed"])
            self.assertTrue(sdk.state.snapshot().in_hold)
        finally:
            motion.close()
            sdk.close()

    def test_place_motion_allows_relative_speed_explained_by_command(self) -> None:
        """受控下降中的接触点运动不能被立即误判为箱体脱手。"""

        sdk = self._sdk(defer_commands=True)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        tool_refs = (
            "component://tool/left",
            "component://tool/right",
        )
        try:
            self._engage_bilateral_tools(sdk)
            state = sdk.state.snapshot()
            targets = []
            for side in ("left", "right"):
                current = state.end_effectors[side]
                targets.append(
                    {
                        "tool_ref": f"component://tool/{side}",
                        "target_pose": {
                            "frame_id": current.frame_id,
                            "position_m": [
                                current.position[0],
                                current.position[1],
                                current.position[2] - 0.01,
                            ],
                            "orientation_xyzw": list(current.quaternion_xyzw),
                            "observed_at": datetime.now().astimezone().isoformat(),
                            "revision": "place-controlled-motion",
                        },
                    }
                )
            started = motion.invoke(
                "MoveEndEffector",
                {
                    "invocation_id": "place-controlled-motion",
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "target_ref": "slot://pallet-b/cell-1",
                    "target_revision": "slot-r1",
                    "targets": targets,
                    "coordination": "synchronized",
                    "purpose": "place",
                    "expected_object_pose": POSE,
                    "maximum_speed_mps": 0.04,
                    "max_contact_force_n": 60.0,
                },
            )
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)

            moving_tools = dict(state.tool_states)
            for tool_ref in tool_refs:
                moving_tools[tool_ref] = moving_tools[tool_ref].model_copy(
                    update={"hook_tangential_speed_m_s": 0.04}
                )
            moving_state = state.model_copy(update={"tool_states": moving_tools})
            with (
                patch.object(sdk.state, "snapshot", return_value=moving_state),
                patch(
                    "r1pro_abilities.handlers.manipulator_motion.load_failure_requires_stop",
                    return_value=True,
                ),
            ):
                refreshed = motion.invoke(
                    "GetExecution", {"invocation_id": "place-controlled-motion"}
                )

            self.assertEqual(refreshed["status"], ExecutionStatus.RUNNING.value)
            value = refreshed["observations"][-1]["value"]
            self.assertTrue(value["stable_bilateral_load"])
            self.assertFalse(value["slip_detected"])
            self.assertEqual(
                value["tool_states"][tool_refs[0]]["hook_tangential_speed_m_s"],
                0.04,
            )
        finally:
            motion.close()
            sdk.close()

    def test_support_transfer_uses_slot_xy_and_external_contact(self) -> None:
        """落座支撑按目标区域和真实接触判断，不增加毫米级高度门限。"""

        sdk = self._sdk()
        expected = {
            "frame_id": "world",
            "position_m": [1.2, 1.5, 0.585],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        snapshot = SceneSnapshot(
            scene_key="depalletizing",
            instance_id="scene-1",
            generation=1,
            coordinate_frame="world",
            objects=[
                SceneObject(
                    source_id="tote-1",
                    category="tote",
                    name="tote-1",
                    pose=Pose(
                        frame_id="world",
                        position=(1.225, 1.5, 0.62415),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(0.6, 0.4, 0.34),
                    state={"in_contact": True},
                )
            ],
            regions=[
                SceneRegion(
                    source_id="slot-1",
                    name="slot-1",
                    pose=Pose(
                        frame_id="world",
                        position=(1.2, 1.5, 0.415),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(0.56, 0.36, 0.02),
                )
            ],
            observed_at=datetime.fromisoformat("2026-08-26T00:00:00+00:00"),
        )
        handler = SimpleNamespace(
            sdk=sdk,
            _external_object_contacts=lambda _object_ref: ["pallet-b"],
        )
        try:
            with patch.object(sdk.state, "scene_snapshot", return_value=snapshot):
                for purpose in ("place", "unseat", "disengage", "clearance"):
                    record = SimpleNamespace(
                        request={
                            "purpose": purpose,
                            "object_ref": "tote-1",
                            "target_ref": "slot-1",
                            "expected_object_pose": expected,
                        }
                    )
                    confirmed, details = (
                        ManipulatorMotionHandler._placement_support_transfer(
                            handler, record
                        )
                    )
                    self.assertTrue(confirmed)
                tilted_object = snapshot.objects[0].model_copy(update={
                    "pose": snapshot.objects[0].pose.model_copy(update={
                        # 绕水平轴约29度：仍在目标Region且接触托盘，但已经
                        # 明显斜靠，不能作为释放夹具的支撑转移证据。
                        "quaternion_xyzw": (0.24740396, 0.0, 0.0, 0.96891242),
                    }),
                })
                tilted_snapshot = snapshot.model_copy(update={
                    "objects": [tilted_object],
                })
                with patch.object(
                    sdk.state, "scene_snapshot", return_value=tilted_snapshot
                ):
                    tilted_confirmed, tilted_details = (
                        ManipulatorMotionHandler._placement_support_transfer(
                            handler, record
                        )
                    )
                self.assertFalse(tilted_confirmed)
                self.assertGreater(
                    tilted_details["orientation_error_rad"],
                    tilted_details["orientation_tolerance_rad"],
                )
        finally:
            sdk.close()

        assert details is not None
        # 39.15 mm 超过旧实现按箱高计算的 34 mm 阈值，但对象已在目标
        # 区域并接触托盘，应认定为正常支撑转移而不是空中失载。
        self.assertAlmostEqual(details["vertical_error_m"], 0.03915)
        self.assertNotIn("vertical_tolerance_m", details)
        self.assertAlmostEqual(details["orientation_error_rad"], 0.0)
        self.assertTrue(details["within_target_xy"])
        self.assertEqual(details["external_contacts"], ["pallet-b"])

    def test_support_transfer_excludes_robot_tool_contacts(self) -> None:
        """夹具自身接触不能冒充目标支撑；外部接触保留公开对象身份。"""

        sdk = self._sdk()
        handler = SimpleNamespace(sdk=sdk)
        frame = SimpleNamespace(
            payload={
                "contacts": [
                    {"first": "tote-1", "second": "r1pro-fake-001:left_tool"},
                    {"first": "pallet-b", "second": "tote-1"},
                ]
            }
        )
        try:
            with patch.object(sdk.sensors, "latest", return_value=frame):
                contacts = ManipulatorMotionHandler._external_object_contacts(
                    handler, "tote-1"
                )
        finally:
            sdk.close()

        self.assertEqual(contacts, ["pallet-b"])

    def test_single_tool_place_accepts_live_support_transfer_at_terminal(self) -> None:
        """单侧最终下降到支撑面后，工具自然卸载不能覆盖成功终态。"""

        sdk = self._sdk(defer_commands=True)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        try:
            self._engage_bilateral_tools(sdk)
            state = sdk.state.snapshot()
            left_ref = "component://tool/left"
            current = state.end_effectors["left"]
            details = {
                "object_ref": "object://box-1",
                "target_ref": "slot://pallet-b/cell-1",
                "position_error_m": 0.002,
                "within_target_xy": True,
                "contact": True,
                "scene_generation": 1,
            }
            started = motion.invoke(
                "MoveEndEffector",
                {
                    "invocation_id": "single-tool-place-support-transfer",
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "target_ref": "slot://pallet-b/cell-1",
                    "target_revision": "slot-r1",
                    "targets": [
                        {
                            "tool_ref": left_ref,
                            "target_pose": {
                                "frame_id": current.frame_id,
                                "position_m": list(current.position),
                                "orientation_xyzw": list(current.quaternion_xyzw),
                            },
                        }
                    ],
                    "coordination": "synchronized",
                    "purpose": "place",
                    "expected_object_pose": POSE,
                    "maximum_speed_mps": 0.04,
                    "max_contact_force_n": 60.0,
                    "required_contact_tools": [left_ref],
                },
            )
            sdk.backend.complete_command(started["command_id"])
            release_id = "release-single-place-tool"
            sdk.end_effector.release("left", command_id=release_id)
            sdk.backend.complete_command(release_id)
            with patch.object(
                ManipulatorMotionHandler,
                "_placement_support_transfer",
                return_value=(True, details),
            ):
                completed = motion.invoke(
                    "GetExecution",
                    {"invocation_id": "single-tool-place-support-transfer"},
                )

            self.assertEqual(completed["status"], ExecutionStatus.SUCCEEDED.value)
            value = completed["observations"][-1]["value"]
            self.assertFalse(value["contact_maintained"])
            self.assertTrue(value["support_transfer_confirmed"])
            self.assertFalse(sdk.state.snapshot().in_hold)
        finally:
            motion.close()
            sdk.close()

    def test_place_motion_accepts_load_transfer_only_after_live_support_confirmation(
        self,
    ) -> None:
        """箱体已落到目标支撑面时，单侧卸载不应被当作提前脱手。"""

        sdk = self._sdk(defer_commands=True)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        try:
            self._engage_bilateral_tools(sdk)
            state = sdk.state.snapshot()
            targets = []
            for side in ("left", "right"):
                current = state.end_effectors[side]
                targets.append(
                    {
                        "tool_ref": f"component://tool/{side}",
                        "target_pose": {
                            "frame_id": current.frame_id,
                            "position_m": list(current.position),
                            "orientation_xyzw": list(current.quaternion_xyzw),
                            "revision": "support-transfer",
                        },
                    }
                )
            started = motion.invoke(
                "MoveEndEffector",
                {
                    "invocation_id": "place-support-transfer",
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "target_ref": "slot://pallet-b/cell-1",
                    "target_revision": "slot-r1",
                    "targets": targets,
                    "coordination": "synchronized",
                    "purpose": "place",
                    "expected_object_pose": POSE,
                    "maximum_speed_mps": 0.04,
                    "max_contact_force_n": 60.0,
                },
            )
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)

            release_id = "test-support-transfer-right"
            sdk.end_effector.release("right", command_id=release_id)
            sdk.backend.complete_command(release_id)
            details = {
                "object_ref": "object://box-1",
                "target_ref": "slot://pallet-b/cell-1",
                "position_error_m": 0.001,
                "within_target_xy": True,
                "contact": True,
                "scene_generation": 1,
            }
            with patch(
                "r1pro_abilities.handlers.manipulator_motion."
                "ManipulatorMotionHandler._placement_support_transfer",
                return_value=(True, details),
            ):
                refreshed = motion.invoke(
                    "GetExecution", {"invocation_id": "place-support-transfer"}
                )

            self.assertEqual(refreshed["status"], ExecutionStatus.RUNNING.value)
            value = refreshed["observations"][-1]["value"]
            self.assertTrue(value["support_transfer_confirmed"])
            self.assertEqual(value["support_transfer"], details)
            self.assertFalse(sdk.state.snapshot().in_hold)
        finally:
            motion.close()
            sdk.close()

    def test_lift_waits_for_ability_stability_window_before_starting(self) -> None:
        sdk = self._sdk(defer_commands=True)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        try:
            self._engage_bilateral_tools(sdk)
            stable_state = sdk.state.snapshot()
            right_ref = "component://tool/right"
            transient_tools = dict(stable_state.tool_states)
            transient_tools[right_ref] = transient_tools[right_ref].model_copy(
                update={"hook_contact": False, "hook_force_n": 0.0}
            )
            transient_state = stable_state.model_copy(
                update={"tool_states": transient_tools}
            )
            samples = iter((transient_state,))

            def snapshot():
                return next(samples, stable_state)

            with patch.object(sdk.state, "snapshot", side_effect=snapshot):
                started = motion.invoke(
                    "LiftHeldObject",
                    {
                        "invocation_id": "lift-waits-for-stability",
                        "robot_id": "r1pro-fake-001",
                        "object_ref": "object://box-1",
                        "tools": [
                            "component://tool/left",
                            "component://tool/right",
                        ],
                        "candidate_id": "candidate-bilateral",
                        "distance_m": 0.08,
                        "maximum_speed_mps": 0.05,
                    },
                )

            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)
            self.assertIsNone(started["error"])
        finally:
            motion.close()
            sdk.close()

    def test_lift_replans_one_shorter_segment_before_command(self) -> None:
        """首次IK失败且尚无物理动作时，只缩短当前抬升段一次。"""

        sdk = self._sdk(defer_commands=False)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        try:
            self._engage_bilateral_tools(sdk)
            real_move = sdk.upper_body.move_end_effectors
            attempted_heights: list[float] = []

            def move_with_shorter_retry(targets, **kwargs):
                left = targets["left"]
                current = sdk.state.snapshot().end_effectors["left"]
                attempted_heights.append(left.position[2] - current.position[2])
                if len(attempted_heights) == 1:
                    raise PlanningError("边界姿态的首段IK未收敛")
                return real_move(targets, **kwargs)

            with patch.object(
                sdk.upper_body,
                "move_end_effectors",
                side_effect=move_with_shorter_retry,
            ):
                lifted = motion.invoke(
                    "LiftHeldObject",
                    {
                        "invocation_id": "lift-shorter-segment",
                        "robot_id": "r1pro-fake-001",
                        "object_ref": "object://box-1",
                        "tools": [
                            "component://tool/left",
                            "component://tool/right",
                        ],
                        "candidate_id": "candidate-bilateral",
                        "distance_m": 0.04,
                        "maximum_speed_mps": 0.05,
                    },
                )

            self.assertEqual(lifted["status"], ExecutionStatus.SUCCEEDED.value)
            self.assertEqual(len(attempted_heights), 2)
            self.assertAlmostEqual(attempted_heights[0], 0.04)
            self.assertAlmostEqual(attempted_heights[1], 0.02)
            self.assertAlmostEqual(lifted["result"]["lift_height_m"], 0.02)
            self.assertAlmostEqual(
                lifted["observations"][0]["value"]["requested_lift_height_m"],
                0.02,
            )
        finally:
            motion.close()
            sdk.close()

    def test_lift_defers_load_transfer_motion_to_in_action_monitoring(self) -> None:
        """静止箱体尚未压实竖直支撑时，起升监视负责判断载荷接管。"""
        sdk = self._sdk(defer_commands=True)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        try:
            self._engage_bilateral_tools(sdk)
            state = sdk.state.snapshot()
            left_ref = "component://tool/left"
            moving_tools = dict(state.tool_states)
            moving_tools[left_ref] = moving_tools[left_ref].model_copy(
                update={
                    "hook_support_ratio": 0.0,
                    "hook_tangential_speed_m_s": 0.2,
                }
            )
            moving_state = state.model_copy(update={"tool_states": moving_tools})

            with patch.object(sdk.state, "snapshot", return_value=moving_state):
                started = motion.invoke(
                    "LiftHeldObject",
                    {
                        "invocation_id": "lift-starts-load-transfer",
                        "robot_id": "r1pro-fake-001",
                        "object_ref": "object://box-1",
                        "tools": [
                            "component://tool/left",
                            "component://tool/right",
                        ],
                        "candidate_id": "candidate-bilateral",
                        "distance_m": 0.08,
                        "maximum_speed_mps": 0.05,
                    },
                )

            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)
            self.assertIsNone(started["error"])
        finally:
            motion.close()
            sdk.close()

    def test_lift_rejects_missing_hook_before_start(self) -> None:
        sdk = self._sdk(defer_commands=True)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        try:
            self._engage_bilateral_tools(sdk)
            state = sdk.state.snapshot()
            left_ref = "component://tool/left"
            unsafe_tools = dict(state.tool_states)
            unsafe_tools[left_ref] = unsafe_tools[left_ref].model_copy(
                update={"hook_contact": False, "hook_force_n": 0.0}
            )
            unsafe_state = state.model_copy(update={"tool_states": unsafe_tools})

            with patch.object(sdk.state, "snapshot", return_value=unsafe_state):
                started = motion.invoke(
                    "LiftHeldObject",
                    {
                        "invocation_id": "lift-rejects-missing-hook",
                        "robot_id": "r1pro-fake-001",
                        "object_ref": "object://box-1",
                        "tools": [
                            "component://tool/left",
                            "component://tool/right",
                        ],
                        "candidate_id": "candidate-bilateral",
                        "distance_m": 0.08,
                        "maximum_speed_mps": 0.05,
                    },
                )

            self.assertEqual(started["status"], ExecutionStatus.FAILED.value)
            self.assertEqual(started["error"]["code"], "LOAD_NOT_STABLE")
        finally:
            motion.close()
            sdk.close()

    def test_lift_tolerates_one_transient_contact_sample(self) -> None:
        sdk = self._sdk(defer_commands=True)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        try:
            self._engage_bilateral_tools(sdk)
            started = motion.invoke(
                "LiftHeldObject",
                {
                    "invocation_id": "lift-contact-debounce",
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "tools": [
                        "component://tool/left",
                        "component://tool/right",
                    ],
                    "candidate_id": "candidate-bilateral",
                    "distance_m": 0.08,
                    "maximum_speed_mps": 0.05,
                },
            )
            self.assertEqual(started["status"], ExecutionStatus.RUNNING.value)

            # 运动开始后的单次传感采样可能丢失接触。当前Observation如实
            # 记录异常，但在异常没有持续到时间边界前不阻塞或停止动作。
            left_ref = "component://tool/left"
            stable_state = sdk.state.snapshot()
            transient_tools = dict(stable_state.tool_states)
            transient_tools[left_ref] = transient_tools[left_ref].model_copy(
                update={
                    "hook_contact": False,
                    "clamp_contact": False,
                    "hook_force_n": 0.0,
                    "clamp_force_n": 0.0,
                }
            )
            transient_state = stable_state.model_copy(
                update={"tool_states": transient_tools}
            )
            with patch.object(sdk.state, "snapshot", return_value=transient_state):
                refreshed = motion.invoke(
                    "GetExecution", {"invocation_id": "lift-contact-debounce"}
                )

            self.assertEqual(refreshed["status"], ExecutionStatus.RUNNING.value)
            transient_value = refreshed["observations"][-1]["value"]
            # stable_load是Ability基于真实时间窗口给出的动作结论；同一条
            # Observation仍保留本次原始接触缺失，便于诊断而不驱动误停。
            self.assertTrue(transient_value["stable_load"])
            self.assertFalse(transient_value["tool_states"][left_ref]["hook_contact"])

            with patch.object(sdk.state, "snapshot", return_value=stable_state):
                recovered = motion.invoke(
                    "GetExecution", {"invocation_id": "lift-contact-debounce"}
                )
            self.assertEqual(recovered["status"], ExecutionStatus.RUNNING.value)
            self.assertTrue(recovered["observations"][-1]["value"]["stable_load"])
        finally:
            motion.close()
            sdk.close()

    def test_lift_load_transfer_keeps_raw_motion_without_false_stop(self) -> None:
        """托盘向夹具转移载荷时，相对运动不应在接触完整时冒充脱手。"""

        sdk = self._sdk(defer_commands=True)
        motion = R1ProAbilityService(AbilityRole.MANIPULATOR_MOTION, sdk)
        tool_refs = (
            "component://tool/left",
            "component://tool/right",
        )
        try:
            self._engage_bilateral_tools(sdk)
            started = motion.invoke(
                "LiftHeldObject",
                {
                    "invocation_id": "lift-load-transfer",
                    "robot_id": "r1pro-fake-001",
                    "object_ref": "object://box-1",
                    "tools": list(tool_refs),
                    "candidate_id": "candidate-bilateral",
                    "distance_m": 0.08,
                    "maximum_speed_mps": 0.05,
                },
            )
            state = sdk.state.snapshot()
            moving_tools = dict(state.tool_states)
            moving_tools[tool_refs[0]] = moving_tools[tool_refs[0]].model_copy(
                update={"hook_tangential_speed_m_s": 0.05}
            )
            moving_state = state.model_copy(update={"tool_states": moving_tools})
            with patch.object(sdk.state, "snapshot", return_value=moving_state):
                refreshed = motion.invoke(
                    "GetExecution", {"invocation_id": "lift-load-transfer"}
                )

            self.assertEqual(refreshed["status"], ExecutionStatus.RUNNING.value)
            value = refreshed["observations"][-1]["value"]
            self.assertTrue(value["stable_load"])
            self.assertFalse(value["slip_detected"])
            self.assertEqual(
                value["tool_states"][tool_refs[0]]["hook_tangential_speed_m_s"],
                0.05,
            )
        finally:
            motion.close()
            sdk.close()

    def test_verify_arrival_refreshes_held_object_evidence(self) -> None:
        sdk = self._sdk(defer_commands=False)
        service = R1ProAbilityService(AbilityRole.NAVIGATION, sdk)
        held = {
            "object_ref": "object://box-1",
            "robot_ref": "r1pro-fake-001",
            "tool_refs": [
                "component://tool/left",
                "component://tool/right",
            ],
            "grasp_pose": POSE,
            "object_pose": POSE,
            "object_size_m": [0.3, 0.2, 0.15],
            "grasp_candidate_id": "candidate-1",
            "grasp_confidence": 0.97,
            "estimated_mass_kg": 1.0,
            "verified_at": "2026-08-09T00:00:00+00:00",
            "robot_state_revision": "robot-state-before-navigation",
            "evidence_refs": ["evidence://grasp/verified"],
        }
        try:
            self._engage_bilateral_tools(sdk)
            source_observed_at = sdk.state.snapshot().observed_at
            verified = service.invoke(
                "VerifyArrival",
                {
                    "robot_id": "r1pro-fake-001",
                    "target": {
                        **TARGET,
                        "pose": {
                            **POSE,
                            "position_m": [0.0, 0.0, 0.0],
                            "revision": "arrival-target-r1",
                        },
                    },
                    "navigation_purpose": "carry_to_place",
                    "arrival_radius_m": 0.5,
                    "require_visual_confirmation": True,
                    "carrying_object": held,
                },
            )
            refreshed = verified["result"]["carrying_object"]
            self.assertEqual(refreshed["object_ref"], held["object_ref"])
            self.assertNotEqual(
                refreshed["robot_state_revision"], held["robot_state_revision"]
            )
            live_state = sdk.state.snapshot()
            self.assertEqual(
                refreshed["tool_poses"]["component://tool/left"]["position_m"],
                list(live_state.end_effectors["left"].position),
            )
            self.assertEqual(
                refreshed["tool_poses"]["component://tool/right"]["position_m"],
                list(live_state.end_effectors["right"].position),
            )
            self.assertIn("arrival-held", refreshed["evidence_refs"][-1])
            refreshed_at = datetime.fromisoformat(refreshed["verified_at"])
            self.assertGreater(refreshed_at, source_observed_at)
            self.assertEqual(
                verified["observations"][0]["kind"], "manipulation.held_object"
            )
        finally:
            service.close()
            sdk.close()

    def test_placement_lateral_clearance_only_uses_same_level_neighbor(self) -> None:
        """净空观测忽略下层支撑箱和另一行箱体，只返回同层侧向间隙。"""

        moving = SceneObject(
            source_id="moving",
            category="tote",
            name="moving",
            pose=Pose(
                frame_id="world",
                position=(0.0, 0.0, 1.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            extent=(0.6, 0.4, 0.34),
        )
        snapshot = SceneSnapshot(
            scene_key="layout001",
            instance_id="scene-1",
            generation=1,
            coordinate_frame="world",
            observed_at=datetime.now().astimezone(),
            objects=[
                moving,
                SceneObject(
                    source_id="left-neighbor",
                    category="tote",
                    name="left",
                    pose=Pose(
                        frame_id="world",
                        position=(1.19, 0.29, 0.66),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(0.6, 0.4, 0.34),
                ),
                SceneObject(
                    source_id="other-row",
                    category="tote",
                    name="row",
                    pose=Pose(
                        frame_id="world",
                        position=(1.81, 0.71, 0.66),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(0.6, 0.4, 0.34),
                ),
                SceneObject(
                    source_id="lower-layer",
                    category="tote",
                    name="lower",
                    pose=Pose(
                        frame_id="world",
                        position=(1.81, 0.29, 0.32),
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    extent=(0.6, 0.4, 0.34),
                ),
            ],
            regions=[],
        )
        clearance = _placement_lateral_clearances(
            snapshot,
            moving,
            {
                "position_m": [1.81, 0.29, 0.66],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        )

        self.assertAlmostEqual(clearance["negative"], 0.02, places=6)
        self.assertIsNone(clearance["positive"])

    def test_map_reference_is_resolved_against_live_scene_hint(self) -> None:
        items = [
            SceneObject(
                source_id="tote-left",
                category="tote",
                name="left",
                pose=Pose(
                    frame_id="world",
                    position=(0.2, 0.0, 0.4),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                extent=(0.6, 0.4, 0.34),
            ),
            SceneObject(
                source_id="tote-right",
                category="tote",
                name="right",
                pose=Pose(
                    frame_id="world",
                    position=(1.2, 0.0, 0.4),
                    quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
                extent=(0.6, 0.4, 0.34),
            ),
        ]

        selected = _resolve(
            items,
            "entity-uuid-from-semantic-map",
            pose_hint={"position_m": [1.18, 0.0, 0.4]},
            extent_hint=[0.6, 0.4, 0.34],
            category_hint="tote",
        )

        self.assertEqual(selected.source_id, "tote-right")

    def test_model_uses_default_or_allowed_override(self) -> None:
        sdk = self._sdk()
        models = ModelProfileRegistry.load(ROOT / "configs/models.example.json")
        service = R1ProAbilityService(AbilityRole.OBJECT_PERCEPTION, sdk, models)
        request = {
            "object_ref": "object://box-1",
            "minimum_confidence": 0.65,
        }
        try:
            default = service.invoke("LocateObject", request)
            alternate = service.invoke(
                "LocateObject", {**request, "model_profile": "fake-perception-alt"}
            )
            self.assertEqual(
                default["result"]["model_profile"], "fake-perception-default"
            )
            self.assertEqual(
                alternate["result"]["model_profile"], "fake-perception-alt"
            )
        finally:
            service.close()
            sdk.close()

    def test_model_from_another_role_and_deployment_fields_are_rejected(self) -> None:
        sdk = self._sdk()
        models = ModelProfileRegistry.load(ROOT / "configs/models.example.json")
        service = R1ProAbilityService(AbilityRole.OBJECT_PERCEPTION, sdk, models)
        request = {"object_ref": "object://box-1", "minimum_confidence": 0.65}
        try:
            with self.assertRaises(ModelProfileError):
                service.invoke(
                    "LocateObject", {**request, "model_profile": "fake-grasp-default"}
                )
            with self.assertRaises(ValidationError):
                service.invoke("LocateObject", {**request, "robot_ip": "192.0.2.1"})
            with self.assertRaises(ValueError):
                service.invoke("LocateObject", {**request, "robot_id": "another-robot"})
        finally:
            service.close()
            sdk.close()


if __name__ == "__main__":
    unittest.main()
