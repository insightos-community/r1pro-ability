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

"""EndEffector Ability：按稳定 tool_ref 控制一侧或双侧周转箱夹具。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from ..ability_utils import component_name, observation
from ..execution import ExecutionRecord, ExecutionStatus
from ..tool_load import tools_are_unloaded
from .base import AbilityHandler


class EndEffectorHandler(AbilityHandler):

    def execute(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
    ) -> ExecutionRecord:
        if task_name == "HoldObject":
            return self._hold(data, invocation_id, digest)
        if task_name not in {"SetOpening", "CloseUntilContact", "Release"}:
            raise ValueError(f"EndEffector 不支持 Task: {task_name}")

        resources = []
        prepared = []
        for index, tool in enumerate(data["tools"]):
            side = component_name(self.sdk, str(tool["tool_ref"]), "end_effector")
            resources.append(f"gripper:{side}")
            prepared.append((index, side, tool))

        def start_tool(item):
            index, side, tool = item
            command_id = f"{invocation_id}:tool:{index}"
            if task_name == "CloseUntilContact":
                return self.sdk.end_effector.close_until_contact(
                    side,
                    float(tool["target_position_m"]),
                    command_id=command_id,
                    force_limit_n=float(tool["maximum_force_n"]),
                )
            # SetOpening 与 Release 共用位置控制，完成后的语义证据分别收敛。
            return self.sdk.end_effector.set_opening(
                side,
                float(tool["target_position_m"]),
                command_id=command_id,
                force_limit_n=float(tool["maximum_force_n"]),
            )

        # Runtime 当前每次只接受一个工具命令，而且 HTTP 调用会等待该命令终态。
        # 若在普通 for 循环中调用，左侧闭合完成后右侧才会启动，自由箱体会被
        # 单侧推离已经建立的钩接触。这里仅对同一个语义动作中的一至两个工具
        # 并发下发；Runtime 仍分别执行真实命令，Ability 再统一收敛、反馈和停止。
        with ThreadPoolExecutor(max_workers=len(prepared)) as executor:
            commands = tuple(executor.map(start_tool, prepared))
        return self.register_commands(
            commands,
            task_name,
            data,
            invocation_id,
            digest,
            resource=",".join(resources),
            result={"tool_refs": [item["tool_ref"] for item in data["tools"]]},
        )

    def after_refresh(self, record: ExecutionRecord) -> ExecutionRecord:
        if record.status is not ExecutionStatus.SUCCEEDED:
            return record
        state = self.sdk.state.snapshot()
        # HoldObject 是同步形成的终态安全证据，没有底层位置命令需要再次收敛。
        # 如果继续落入下方 Release 分支，下一次 GetExecution 会把正确的 hold
        # 回执覆盖成 RELEASE_NOT_CONFIRMED，导致 Skill 和 Web 都看到伪失败。
        if record.task_name == "HoldObject":
            return record

        tool_states = {
            str(item["tool_ref"]): (
                state.tool_states.get(str(item["tool_ref"]))
            )
            for item in record.request.get("tools", [])
        }
        values = {
            key: value.model_dump(mode="json") if value is not None else None
            for key, value in tool_states.items()
        }
        all_tool_states = {
            key: value.model_dump(mode="json")
            for key, value in state.tool_states.items()
        }
        if record.task_name == "SetOpening":
            requested = {
                str(item["tool_ref"]): float(item["target_position_m"])
                for item in record.request["tools"]
            }
            reached = bool(values) and all(
                value is not None
                and abs(float(value["position"]) - requested[key]) <= 0.002
                and not value["sensor_fault"]
                for key, value in values.items()
            )
            obs = observation(
                "manipulation.tool_opening",
                f"robot-sdk://{self.sdk.robot_id}/end-effector",
                self.sdk.robot_id,
                {
                    "tools": values,
                    "requested_positions_m": requested,
                    "reached": reached,
                },
                revision=record.request_digest[:16],
                confidence=1.0,
            )
            return self.executions.update_observations(
                record.invocation_id,
                (obs,),
                {"tool_states": values, "verified": reached},
                status=ExecutionStatus.SUCCEEDED if reached else ExecutionStatus.FAILED,
                error=None
                if reached
                else {
                    "code": "OPENING_NOT_REACHED",
                    "message": "工具没有到达请求的预抓取开度",
                },
            )
        if record.task_name == "CloseUntilContact":
            expected = str(record.request["object_ref"])
            # 闭合只负责把上夹片送到候选声明的位置。下钩已经水平进入凹槽
            # 并向上seat；在闭合Action里再要求完整持续承载会形成循环依赖：
            # 不先轻微抬升就没有承载，而旧实现又因“没有承载”拒绝抬升。
            # 后续 seat 动作和 RobotState.VerifyToolLoad 才负责确认真实承载。
            closure_completed = bool(values) and all(
                value is not None
                and not value["sensor_fault"]
                and (
                    bool(value["reached_target"])
                    or bool(value["clamp_contact"])
                )
                for key, value in values.items()
            )
            kind = "manipulation.tool_closed"
            payload = {
                "candidate_id": record.request["candidate_id"],
                "object_ref": expected,
                "tools": values,
                "closure_completed": closure_completed,
                "all_tool_states": all_tool_states,
            }
        else:
            kind = "manipulation.object_released"
            descriptors = self.sdk.state.capabilities().tools
            selected_refs = list(values)
            # 工具打开后仍可能与箱沿发生无载擦碰。Release 的物理语义是
            # “指定工具不再承载物体”，而不是要求所有接触传感器绝对归零。
            # 原始接触、力和方向仍完整保存在 tools/all_tool_states 中，便于
            # Web Inspector 和后续故障分析；这里不引入箱型特定力阈值。
            released = tools_are_unloaded(state, descriptors, selected_refs)
            all_released = tools_are_unloaded(
                state, descriptors, list(state.tool_states)
            )
            payload = {
                "object_ref": record.request["object_ref"],
                "tools": values,
                "released": released,
                "released_tool_refs": list(values),
                "gripper_empty": all_released,
                "all_tool_states": all_tool_states,
            }
        obs = observation(
            kind,
            f"robot-sdk://{self.sdk.robot_id}/end-effector",
            str(record.request["object_ref"]),
            payload,
            revision=record.request_digest[:16],
            confidence=1.0,
        )
        verified = (
            closure_completed
            if record.task_name == "CloseUntilContact"
            else released
        )
        # 底层位置命令完成只说明执行器停止运动；CloseUntilContact 和 Release
        # 还必须由实时工具状态证明其语义结果。这里直接收敛原 Execution，避免
        # Skill 把一个带 false Observation 的 succeeded 当成真实抓取或释放。
        return self.executions.update_observations(
            record.invocation_id,
            (obs,),
            {"tool_states": values, "verified": verified},
            status=ExecutionStatus.SUCCEEDED if verified else ExecutionStatus.FAILED,
            error=None
            if verified
            else {
                "code": "CONTACT_NOT_CONFIRMED"
                if record.task_name == "CloseUntilContact"
                else "RELEASE_NOT_CONFIRMED",
                "message": "夹具没有安全完成闭合动作"
                if record.task_name == "CloseUntilContact"
                else "未取得工具完全释放证据",
            },
        )

    def _hold(
        self, data: Mapping[str, Any], invocation_id: str, digest: str
    ) -> ExecutionRecord:
        state = self.sdk.safety.hold()
        confirmed = bool(state.in_hold)
        selected = {
            ref: state.tool_states[ref].model_dump(mode="json")
            for ref in (item["tool_ref"] for item in data.get("tools", []))
            if ref in state.tool_states
        }
        obs = observation(
            "manipulation.tool_hold",
            f"robot-sdk://{self.sdk.robot_id}/safety",
            str(data.get("object_ref") or self.sdk.robot_id),
            {"in_hold": confirmed, "tools": selected},
            revision=digest[:16],
            confidence=1.0 if confirmed else 0.0,
        )
        return self.executions.register_terminal(
            "HoldObject",
            digest,
            ExecutionStatus.SUCCEEDED if confirmed else ExecutionStatus.INTERRUPTED,
            {
                "safe": confirmed,
                "physical_state": "hold" if confirmed else "unknown",
                "stop_evidence": {
                    "holding": confirmed,
                    "reason": data["reason"],
                    "tools": selected,
                },
            },
            invocation_id,
            observations=(obs,),
            error=None
            if confirmed
            else {"code": "HOLD_UNCONFIRMED", "message": "未取得夹具 hold 证据"},
            request=data,
        )
