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

"""RobotState Ability：公开实时 Robot 状态并验证工具承载条件。"""

from __future__ import annotations

from typing import Any, Mapping

from ..ability_utils import observation
from ..execution import ExecutionRecord
from ..tool_load import wait_for_stable_tool_load
from .base import AbilityHandler


class RobotStateHandler(AbilityHandler):
    def execute(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
    ) -> ExecutionRecord:
        if task_name == "GetRobotState":
            return self._get_robot_state(task_name, data, invocation_id, digest)
        if task_name == "VerifyToolLoad":
            return self._verify_tool_load(task_name, data, invocation_id, digest)
        raise ValueError(f"RobotState 不支持 Task: {task_name}")

    def _get_robot_state(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
    ) -> ExecutionRecord:
        state = self.sdk.state.snapshot()
        value = state.model_dump(mode="json")
        value.update(
            {
                "backend": self.sdk.deployment.robot.backend,
                "firmware_profile": self.sdk.deployment.robot.sdk.firmware_profile,
            }
        )
        obs = observation(
            "robot.state",
            f"robot-sdk://{self.sdk.robot_id}/state",
            self.sdk.robot_id,
            value,
            revision=f"robot-state-g{state.generation}",
            frame_id=state.base_pose.frame_id if state.base_pose else None,
        )
        return self.succeeded(
            task_name,
            data,
            invocation_id,
            digest,
            {
                "robot_id": state.robot_id,
                "generation": state.generation,
                "in_hold": state.in_hold,
            },
            (obs,),
        )

    def _verify_tool_load(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
    ) -> ExecutionRecord:
        tool_refs = [str(ref) for ref in data["tool_refs"]]

        # Runtime 和 Robot SDK 只给出每次采样的接触、力与相对速度。工具是否
        # 在当前动作中持续满足承载条件，由 Ability 使用单调真实时间观察；
        # 这里既不读取仿真帧计数，也不保存跨调用的全局 HeldObjectState。
        state, load, observed_duration_ms = wait_for_stable_tool_load(
            self.sdk, tool_refs
        )
        tools: list[dict[str, Any]] = []
        for tool_ref in tool_refs:
            tool_state = load.tool_states.get(tool_ref)
            if tool_state is None:
                tools.append({"tool_ref": tool_ref, "available": False})
                continue
            tools.append(
                {
                    "tool_ref": tool_ref,
                    "available": True,
                    "position": tool_state.position,
                    "velocity": tool_state.velocity,
                    "effort": tool_state.effort,
                    "hook_contact": tool_state.hook_contact,
                    "clamp_contact": tool_state.clamp_contact,
                    "hook_force_n": tool_state.hook_force_n,
                    "clamp_force_n": tool_state.clamp_force_n,
                    "hook_support_ratio": tool_state.hook_support_ratio,
                    "relative_tangential_speed_mps": (
                        tool_state.hook_tangential_speed_m_s
                    ),
                    "sensor_fault": tool_state.sensor_fault,
                }
            )
        result = {
            "condition_satisfied": load.safe,
            "observed_duration_ms": observed_duration_ms,
            "slip_detected": load.slipping,
            "overload_detected": load.overloaded,
            "sensor_fault": load.sensor_fault,
        }
        obs = observation(
            "robot.tool_load",
            f"robot-sdk://{self.sdk.robot_id}/state",
            self.sdk.robot_id,
            {
                **result,
                "tools": tools,
                "reasons": list(load.reasons),
            },
            revision=f"robot-state-g{state.generation}",
            frame_id=state.base_pose.frame_id if state.base_pose else None,
        )
        return self.succeeded(
            task_name,
            data,
            invocation_id,
            digest,
            result,
            (obs,),
        )
