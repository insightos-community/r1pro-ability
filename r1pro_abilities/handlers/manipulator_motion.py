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

"""ManipulatorMotion Ability：单侧或双侧同步末端短运动。"""

from __future__ import annotations

from dataclasses import replace
import json
from datetime import datetime, timezone
from math import acos, sqrt
from time import monotonic, sleep
from typing import Any, Mapping, Sequence

from semantic_robot_sdk_core import CommandState, PlanningError, Pose

from ..ability_utils import (
    component_name,
    observation,
    pose,
    scene_collision_environment,
)
from ..execution import ExecutionRecord, ExecutionStatus
from ..tool_load import (
    DEFAULT_TOOL_LOAD_POLICY,
    load_failure_requires_stop,
    parse_load_unstable_since,
    read_tool_load,
    wait_for_confirmed_tool_engagement,
)
from .base import AbilityHandler


# 支撑转移只需要排除明显侧翻，不做精确姿态匹配。20度仍覆盖接触沉降和
# 厘米级落座偏差，但不会把箱体斜靠在邻箱/夹具上的瞬时接触当作托盘承重。
MAX_SUPPORT_TRANSFER_ORIENTATION_ERROR_RAD = 0.35


class ManipulatorMotionHandler(AbilityHandler):
    _TARGET_CENTER_FALLBACK_TOLERANCE_M = 0.05

    def after_refresh(self, record: ExecutionRecord) -> ExecutionRecord:
        """在带载末端运动期间核验真实传感状态，失稳时停止并 hold。"""
        if record.task_name == "MoveEndEffector":
            purpose = record.request.get("purpose")
            if purpose in {"insert", "seat"}:
                if record.request.get("required_contact_tools"):
                    record = self._refresh_required_tool_contact(record)
                    if record.status not in {
                        ExecutionStatus.ACCEPTED,
                        ExecutionStatus.RUNNING,
                        ExecutionStatus.SUCCEEDED,
                    }:
                        return record
                # 水平insert的目标位于侧面凹槽内部、上沿下方。此时下钩
                # 可以没有竖直支撑；若把碰到槽口或箱体侧壁当成“插入成功”
                # 提前停下，钩脚会留在槽外，随后只能空行程
                # 压紧。insert 因此必须完成规划终点，真正的接合只在向上 seat
                # 时根据支撑方向和受力确认。
                return (
                    record
                    if purpose == "insert"
                    else self._refresh_seat_contact(record)
                )
            if record.request.get("required_contact_tools"):
                return self._refresh_required_tool_contact(record)
        if record.task_name == "MoveEndEffector" and record.request.get("purpose") in {
            "place",
            "transport",
        }:
            return self._refresh_loaded_motion(record)
        if record.task_name != "LiftHeldObject" or record.status not in {
            ExecutionStatus.ACCEPTED,
            ExecutionStatus.RUNNING,
            ExecutionStatus.SUCCEEDED,
        }:
            return record

        expected_object = str(record.request["object_ref"])
        tool_refs = [str(value) for value in record.request["tools"]]
        state, load = read_tool_load(self.sdk, tool_refs)
        unstable_since = parse_load_unstable_since(
            record.result.get("_load_unstable_since")
        )
        result: dict[str, Any] = {}
        # 起升最初几毫米是夹具从托盘接管载荷的阶段，接触点必然可能相对
        # 运动。这里不能拿一个通用速度阈值把正常取载误判为脱手；真正的
        # 动作安全边界是左右钩入、压紧、受力方向和传感状态是否仍成立。
        # 箱体最终是否跟随双臂抬起，随后由 VerifyGrasp 比较实时物体位姿，
        # 不能由关节轨迹进度或 Runtime 的仿真实现推断。
        attachment_intact = load.engaged and not (load.overloaded or load.sensor_fault)
        if attachment_intact:
            result["_load_unstable_since"] = None
        elif unstable_since is None:
            unstable_since = datetime.now(timezone.utc)
            result["_load_unstable_since"] = unstable_since.isoformat()
        attachment_failure_confirmed = not attachment_intact and (
            load.sensor_fault
            or load.overloaded
            or (
                unstable_since is not None
                and load_failure_requires_stop(load, unstable_since)
            )
        )
        effective_attachment = not attachment_failure_confirmed
        payload = {
            "candidate_id": record.request["candidate_id"],
            "requested_lift_height_m": record.request["distance_m"],
            # Runtime 命令进度来自实际双臂命令。这里把它换算为当前抬升量，
            # 只用于 Skill 判断流式 Observation 是否完整；最终是否真的抬起
            # 箱体仍由 VerifyGrasp 比较前后物体 Pose，不能拿轨迹进度代替物理证据。
            "lift_height_m": float(record.request["distance_m"]) * record.progress,
            "object_follows_tools": effective_attachment,
            "stable_load": effective_attachment,
            # 取载阶段的接触点相对运动保留在下方原始 tool_states 中；只有
            # 已确认的承载丢失才形成会驱动 Skill 停止的语义结论。
            "slip_detected": False,
            "overloaded": load.overloaded,
            "sensor_fault": load.sensor_fault,
            "reasons": list(load.reasons),
            "tool_states": load.value()["tool_states"],
        }
        obs = observation(
            "lift_progress",
            f"robot-sdk://{self.sdk.robot_id}/state",
            expected_object,
            payload,
            revision=f"lift-{state.generation}-{state.observed_at.isoformat()}",
            confidence=1.0,
        )

        record = self.executions.update_observations(
            record.invocation_id, (obs,), result=result
        )
        # 任一侧异常持续超过动作安全窗口时，不能继续由另一侧拖住箱体。
        # 单次速度尖峰只记录Observation；Runtime不产生稳定/滑移业务结论。
        if attachment_failure_confirmed:
            hold_confirmed = False
            if record.status in {ExecutionStatus.ACCEPTED, ExecutionStatus.RUNNING}:
                stopped = self.executions.stop(
                    self.sdk, record.invocation_id, "carrying_load_unstable"
                )
                hold_confirmed = stopped.status is ExecutionStatus.STOPPED
                record = stopped
            else:
                hold_confirmed = self.sdk.safety.hold().in_hold
            terminal = (
                ExecutionStatus.FAILED
                if hold_confirmed
                else ExecutionStatus.INTERRUPTED
            )
            return self.executions.update_observations(
                record.invocation_id,
                (obs,),
                {**payload, "hold_confirmed": hold_confirmed},
                status=terminal,
                error={
                    "code": "LOAD_UNSTABLE_DURING_LIFT"
                    if hold_confirmed
                    else "LOAD_STATE_UNKNOWN_DURING_LIFT",
                    "message": "抬升期间承载失稳，Robot 已进入 hold"
                    if hold_confirmed
                    else "抬升期间承载状态未知，无法确认 Robot hold",
                },
            )

        if record.status is ExecutionStatus.SUCCEEDED:
            return self.executions.update_observations(
                record.invocation_id, (obs,), payload
            )
        return record

    def _refresh_seat_contact(self, record: ExecutionRecord) -> ExecutionRecord:
        """让seat由凹槽内真实支撑接触结束，而不是强迫关节穿过机械约束。

        seat的目标位姿贴近凹槽上沿内侧，用来表达运动方向和最大行程；它不是
        可自由到达的几何终点。Ability在运动期间读取真机同样可提供的工具
        力和支撑方向，一旦钩爪形成有效支撑就停止并hold当前位置。这里只用
        当前采样终止接触动作；close后的连续承载仍由VerifyToolLoad负责。
        """

        if record.status not in {
            ExecutionStatus.ACCEPTED,
            ExecutionStatus.RUNNING,
        }:
            return record
        tool_refs = [
            str(item["tool_ref"]) for item in record.request.get("targets", [])
        ]
        state = self.sdk.state.snapshot()
        descriptors = {
            item.tool_ref: item for item in self.sdk.state.capabilities().tools
        }
        reasons: list[str] = []
        tool_states: dict[str, Any] = {}
        position_errors = self._target_position_errors(record, state)
        if not position_errors:
            reasons.append("end_effector_pose_unavailable")
        for tool_ref in tool_refs:
            descriptor = descriptors.get(tool_ref)
            value = state.tool_states.get(tool_ref)
            tool_states[tool_ref] = (
                value.model_dump(mode="json") if value is not None else None
            )
            if descriptor is None or value is None:
                reasons.append(f"{tool_ref}:state_unavailable")
                continue
            if value.sensor_fault:
                reasons.append(f"{tool_ref}:sensor_fault")
            if not value.hook_contact:
                reasons.append(f"{tool_ref}:hook_contact_missing")
            # seat目标表示钩脚在凹槽内向上寻找槽内上表面的最大行程，不要求
            # Robot穿过接触面后仍到达自由空间终点。只要下钩已有真实
            # 正向受力且方向确实能够承重，就应立即停止并闭合上夹片；完整的
            # 力阈值和持续时间留给 close 后的 VerifyToolLoad。否则控制器会
            # 为了追逐不可到达的关节目标持续顶压，最后以关节误差超时结束。
            if value.hook_force_n <= 0.0:
                reasons.append(f"{tool_ref}:hook_force_missing")
            if (
                value.hook_support_ratio
                < DEFAULT_TOOL_LOAD_POLICY.minimum_support_ratio
            ):
                reasons.append(f"{tool_ref}:vertical_support_low")
            if value.hook_force_n > descriptor.maximum_force_n:
                reasons.append(f"{tool_ref}:force_limit_exceeded")
        if reasons:
            return record

        stopped = self.sdk.safety.stop_and_hold(record.command_id)
        holding = self.sdk.state.snapshot().in_hold
        confirmed = (
            stopped.status
            not in {
                CommandState.ACCEPTED,
                CommandState.RUNNING,
                CommandState.INTERRUPTED,
                CommandState.UNKNOWN,
            }
            and holding
        )
        payload = {
            **dict(record.result.get("_success_result") or {}),
            "contact_seated": confirmed,
            "tool_refs": tool_refs,
            "tool_states": tool_states,
            "position_errors_m": position_errors,
            "stop_evidence": {
                "command_status": stopped.status.value,
                "holding": holding,
                "reason": "seat_contact_reached",
            },
        }
        obs = observation(
            "manipulation.seat_contact",
            f"robot-sdk://{self.sdk.robot_id}/state",
            str(record.request.get("object_ref") or self.sdk.robot_id),
            payload,
            revision=f"seat-contact-{state.generation}-{state.observed_at.isoformat()}",
            confidence=1.0,
        )
        return self.executions.update_observations(
            record.invocation_id,
            (obs,),
            payload,
            status=(
                ExecutionStatus.SUCCEEDED if confirmed else ExecutionStatus.INTERRUPTED
            ),
            error=(
                None
                if confirmed
                else {
                    "code": "SEAT_STOP_UNCONFIRMED",
                    "message": "钩脚已贴住凹槽内上表面，但无法确认Robot已停止并进入hold",
                }
            ),
        )

    def _target_position_errors(
        self, record: ExecutionRecord, state: Any
    ) -> dict[str, float]:
        """计算真实末端与本次接触目标的距离，不解释接触属于哪类物体。"""

        errors: dict[str, float] = {}
        for item in record.request.get("targets", []):
            tool_ref = str(item["tool_ref"])
            end_effector = component_name(self.sdk, tool_ref, "manipulator_motion")
            actual = state.end_effectors.get(end_effector)
            if actual is None:
                continue
            expected = tuple(
                float(value) for value in item["target_pose"]["position_m"]
            )
            errors[tool_ref] = sqrt(
                sum(
                    (actual.position[index] - expected[index]) ** 2
                    for index in range(3)
                )
            )
        return errors

    def _refresh_required_tool_contact(
        self, record: ExecutionRecord
    ) -> ExecutionRecord:
        """运动期间保持调用方明确声明的工具接触。

        单侧外拉和第二侧接近时，箱体尚未形成双侧持物状态，但首侧已经产生
        真实物理约束。Ability只根据当前工具传感器和单调时间判断该约束是否
        持续存在；Runtime不输出帧计数或“抓取稳定”业务结论。
        """

        if record.status not in {
            ExecutionStatus.ACCEPTED,
            ExecutionStatus.RUNNING,
            ExecutionStatus.SUCCEEDED,
        }:
            return record
        tool_refs = [str(value) for value in record.request["required_contact_tools"]]
        load_policy = DEFAULT_TOOL_LOAD_POLICY
        if record.request.get("purpose") == "extract":
            # 外拉动作本身会让钩爪与箱体在箱体克服托盘摩擦前产生受控相对
            # 运动。允许范围由本次末端速度与静态滑移阈值共同决定；动作结束
            # 后 Skill 仍会用默认静态窗口重新验证，因此这里不会把持续滑移
            # 误当成稳定抓取，也不依赖 MuJoCo timestep 或箱型经验参数。
            load_policy = replace(
                DEFAULT_TOOL_LOAD_POLICY,
                maximum_tangential_speed_m_s=(
                    DEFAULT_TOOL_LOAD_POLICY.maximum_tangential_speed_m_s
                    + float(record.request["maximum_speed_mps"])
                ),
            )
        state, load = read_tool_load(self.sdk, tool_refs, load_policy)
        contact_maintained, contact_reasons = self._required_contact_state(record, load)
        # 单侧完成最后落座时，承载工具会自然卸载到目标支撑面。这里只接受
        # 当前目标几何、真实接触和物体位置共同证明的支撑转移；空中失载、
        # 过载或传感故障仍然按原安全边界停止，不能靠放宽力阈值掩盖。
        support_transfer, support_details = self._placement_support_transfer(record)
        terminal_contact_failed = False
        if (
            record.status is ExecutionStatus.SUCCEEDED
            and not contact_maintained
            and not support_transfer
            and not (load.overloaded or load.sensor_fault)
        ):
            # 底层关节命令可能恰好在首侧接触丢失的同一次刷新中完成。
            # 如果直接把command=succeeded返回Pilot，Pilot不会再轮询这个
            # Ability Execution，接触丢失便只剩一条Observation而动作仍被
            # 上报成功。终点处用真实时间给传感器一个短恢复窗口；未恢复则
            # 先hold再以失败收敛，不依赖Runtime帧数，也不重放物理动作。
            state, load, contact_maintained, contact_reasons = (
                self._wait_for_terminal_required_contact(record, tool_refs, load_policy)
            )
            support_transfer, support_details = self._placement_support_transfer(record)
            terminal_contact_failed = not contact_maintained and not support_transfer
        unstable_since = parse_load_unstable_since(
            record.result.get("_required_contact_unstable_since")
        )
        result: dict[str, Any] = {}
        if (
            (contact_maintained or support_transfer)
            and not (load.overloaded or load.sensor_fault)
        ):
            result["_required_contact_unstable_since"] = None
        elif unstable_since is None:
            unstable_since = datetime.now(timezone.utc)
            result["_required_contact_unstable_since"] = unstable_since.isoformat()
        payload = {
            "object_ref": record.request.get("object_ref"),
            "required_contact_tools": tool_refs,
            "contact_maintained": contact_maintained,
            "support_transfer_confirmed": support_transfer,
            "support_transfer": support_details,
            "slip_detected": load.slipping,
            "overload_detected": load.overloaded,
            "sensor_fault": load.sensor_fault,
            "reasons": contact_reasons,
            "tool_states": load.value()["tool_states"],
        }
        obs = observation(
            "manipulation.required_tool_contact",
            f"robot-sdk://{self.sdk.robot_id}/state",
            str(record.request.get("object_ref") or self.sdk.robot_id),
            payload,
            revision=f"required-contact-{state.generation}-{state.observed_at.isoformat()}",
            confidence=1.0,
        )
        record = self.executions.update_observations(
            record.invocation_id, (obs,), result=result
        )
        if (
            (
                (not contact_maintained and not support_transfer)
                or load.overloaded
                or load.sensor_fault
            )
            and unstable_since is not None
            and (
                terminal_contact_failed
                or load_failure_requires_stop(load, unstable_since)
            )
        ):
            hold_confirmed = False
            if record.status in {ExecutionStatus.ACCEPTED, ExecutionStatus.RUNNING}:
                stopped = self.executions.stop(
                    self.sdk, record.invocation_id, "required_tool_contact_lost"
                )
                hold_confirmed = stopped.status is ExecutionStatus.STOPPED
                record = stopped
            else:
                hold_confirmed = self.sdk.safety.hold().in_hold
            return self.executions.update_observations(
                record.invocation_id,
                (obs,),
                {**payload, "hold_confirmed": hold_confirmed},
                status=(
                    ExecutionStatus.FAILED
                    if hold_confirmed
                    else ExecutionStatus.INTERRUPTED
                ),
                error={
                    "code": (
                        "REQUIRED_TOOL_CONTACT_LOST"
                        if hold_confirmed
                        else "REQUIRED_TOOL_CONTACT_UNKNOWN"
                    ),
                    "message": (
                        "运动期间首侧工具接触丢失，Robot 已进入 hold"
                        if hold_confirmed
                        else "运动期间首侧工具状态未知，无法确认 Robot hold"
                    ),
                },
            )
        return record

    def _wait_for_terminal_required_contact(
        self,
        record: ExecutionRecord,
        tool_refs: Sequence[str],
        load_policy: Any,
    ) -> tuple[Any, Any, bool, list[str]]:
        """命令终点短暂复核必须维持的接触，避免终态刷新竞态。"""

        deadline = monotonic() + load_policy.acquisition_timeout_s
        stable_since: float | None = None
        while True:
            now = monotonic()
            state, load = read_tool_load(self.sdk, tool_refs, load_policy)
            maintained, reasons = self._required_contact_state(record, load)
            if load.sensor_fault or load.overloaded:
                return state, load, maintained, reasons
            if maintained:
                stable_since = stable_since or now
                if now - stable_since >= load_policy.stable_duration_s:
                    return state, load, True, reasons
            else:
                stable_since = None
            if now >= deadline:
                return state, load, False, reasons
            sleep(load_policy.sample_interval_s)

    @staticmethod
    def _required_contact_state(
        record: ExecutionRecord, load: Any
    ) -> tuple[bool, list[str]]:
        """按当前短动作判断必须维持的物理接触，不伪造稳定持物状态。

        单侧外拉时，下钩脚贴住侧面凹槽的上沿内面牵引箱体，压紧块会因接触顺应性短暂
        卸载；若此时要求静态夹紧力始终不变，正常外拉会被误判失败。外拉仍
        严格要求承重钩接触、支撑方向、受力和传感安全；主动运动的相对速度
        只保留作诊断。动作结束后 Skill 会重新读取实时工具状态，只有下钩仍
        明确接合而压紧接触确实下降时，才允许一次条件式重新夹紧。
        """

        purpose = record.request.get("purpose")
        if purpose in {"clearance", "transfer", "pregrasp", "insert", "seat"}:
            # 外拉后的第二侧接近发生在箱体仍由托盘或下层箱体支撑时。
            # 第一侧此时必须继续钩入、压紧并保持安全受力，但它的钩接触
            # 法向不一定已经转为竖直承载方向；只有双侧接合后的起升动作
            # 才要求稳定承载。把两者混为一谈会让完好的耦合接触被误停。
            ignored_suffixes = (
                ":hook_force_low",
                ":clamp_force_low",
                ":vertical_support_low",
                ":relative_motion_high",
            )
            reasons = [
                reason
                for reason in load.reasons
                if not reason.endswith(ignored_suffixes)
            ]
            maintained = bool(load.tool_states) and all(
                state is not None
                and not state.sensor_fault
                and state.hook_contact
                and state.clamp_contact
                for state in load.tool_states.values()
            )
            maintained = maintained and not (load.overloaded or load.sensor_fault)
            return maintained, reasons
        if purpose != "extract":
            return load.engaged, list(load.reasons)
        # 外拉是在首侧钩爪已经接合后主动让箱体克服支撑摩擦的动作，接触点
        # 沿侧面凹槽产生切向相对运动是这个动作的直接结果，不能把该原始速度
        # 单独解释成“箱体正在脱手”。运动中仍严格检查钩接触、承力方向、
        # 钩力、过载和传感器故障；动作结束后 Skill 会重新压紧，并由
        # VerifyToolLoad 在静止时间窗口内确认完整钩入和压紧。
        ignored_suffixes = (
            ":clamp_contact_missing",
            ":clamp_force_low",
            ":relative_motion_high",
        )
        reasons = [
            reason for reason in load.reasons if not reason.endswith(ignored_suffixes)
        ]
        maintained = bool(load.tool_states) and all(
            state is not None
            and not state.sensor_fault
            and state.hook_contact
            and state.hook_force_n >= DEFAULT_TOOL_LOAD_POLICY.minimum_hook_force_n
            and state.hook_support_ratio
            >= DEFAULT_TOOL_LOAD_POLICY.minimum_support_ratio
            for state in load.tool_states.values()
        )
        maintained = maintained and not (load.overloaded or load.sensor_fault)
        return maintained, reasons

    def _refresh_loaded_motion(self, record: ExecutionRecord) -> ExecutionRecord:
        """携物整理和放置释放前，双侧承载必须在运动中持续成立。"""

        if record.status not in {
            ExecutionStatus.ACCEPTED,
            ExecutionStatus.RUNNING,
            ExecutionStatus.SUCCEEDED,
        }:
            return record
        tool_refs = [str(value["tool_ref"]) for value in record.request["targets"]]
        # 钩爪接触点在双臂受控移动时会沿箱沿产生与指令速度同量级的相对运动，
        # 这不等同于箱体正在从夹具滑落。Ability 用当前动作速度给原始相对速度
        # 建立上界；超过动作可解释范围的相对运动仍按滑移处理。这个判断不依赖
        # MuJoCo 帧数，也不把箱体尺寸或场景经验参数下沉到 Runtime/SDK。
        motion_speed = float(record.request["maximum_speed_mps"])
        load_policy = replace(
            DEFAULT_TOOL_LOAD_POLICY,
            maximum_tangential_speed_m_s=max(
                DEFAULT_TOOL_LOAD_POLICY.maximum_tangential_speed_m_s,
                motion_speed * 1.25,
            ),
        )
        state, load = read_tool_load(self.sdk, tool_refs, load_policy)
        support_transfer, support_details = self._placement_support_transfer(record)
        unstable_since = parse_load_unstable_since(
            record.result.get("_load_unstable_since")
        )
        result: dict[str, Any] = {}
        if load.safe or support_transfer:
            result["_load_unstable_since"] = None
        elif unstable_since is None:
            unstable_since = datetime.now(timezone.utc)
            result["_load_unstable_since"] = unstable_since.isoformat()
        payload = {
            "object_ref": record.request["object_ref"],
            "stable_bilateral_load": load.safe,
            "slip_detected": load.slipping,
            "overload_detected": load.overloaded,
            "sensor_fault": load.sensor_fault,
            "support_transfer_confirmed": support_transfer,
            "support_transfer": support_details,
            "reasons": list(load.reasons),
            "tool_states": load.value()["tool_states"],
        }
        obs = observation(
            "placement.carrying_load",
            f"robot-sdk://{self.sdk.robot_id}/state",
            str(record.request["object_ref"]),
            payload,
            revision=f"loaded-motion-{state.generation}-{state.observed_at.isoformat()}",
            confidence=1.0,
        )
        record = self.executions.update_observations(
            record.invocation_id, (obs,), result=result
        )
        if (
            not load.safe
            and not support_transfer
            and unstable_since is not None
            and load_failure_requires_stop(load, unstable_since, load_policy)
        ):
            hold_confirmed = False
            if record.status in {ExecutionStatus.ACCEPTED, ExecutionStatus.RUNNING}:
                stopped = self.executions.stop(
                    self.sdk, record.invocation_id, "placement_load_unstable"
                )
                hold_confirmed = stopped.status is ExecutionStatus.STOPPED
                record = stopped
            else:
                hold_confirmed = self.sdk.safety.hold().in_hold
            terminal = (
                ExecutionStatus.FAILED
                if hold_confirmed
                else ExecutionStatus.INTERRUPTED
            )
            return self.executions.update_observations(
                record.invocation_id,
                (obs,),
                {**payload, "hold_confirmed": hold_confirmed},
                status=terminal,
                error={
                    "code": "LOAD_UNSTABLE_DURING_PLACEMENT"
                    if hold_confirmed
                    else "LOAD_STATE_UNKNOWN_DURING_PLACEMENT",
                    "message": "放置运动中承载失稳，Robot 已进入 hold"
                    if hold_confirmed
                    else "放置运动中承载状态未知，无法确认 Robot hold",
                },
            )
        return record

    def _placement_support_transfer(
        self, record: ExecutionRecord
    ) -> tuple[bool, dict[str, Any] | None]:
        """确认放置末端的载荷已经从夹具转移到目标支撑面。

        放置最后几毫米内，钩爪受力自然卸载，不能继续用“悬空搬运时必须
        双侧承载”的规则判断。但也不能仅凭轨迹进度放宽安全条件：Ability
        必须重新读取当前 Scene/真机感知，确认物体已经到达本次实时观测得到
        的放置位姿、位于目标区域且存在接触。Runtime 仍只提供对象位姿和
        接触事实，不输出“放置完成”这一业务结论。
        """

        # ``place`` 是双手落座；逐侧释放后的 ``unseat``、``disengage``
        # 和竖直 ``clearance`` 会固定另一侧末端继续退钩。箱体已由托盘
        # 承重时，固定侧可能自然卸载，因此只要调用方同时给出本次
        # expected_object_pose，这几段都复用同一支撑转移事实。普通退钩
        # 没有该位姿，不会被误认为已经放置。
        if record.request.get("purpose") not in {
            "place", "unseat", "disengage", "clearance",
        }:
            return False, None
        expected_value = record.request.get("expected_object_pose")
        object_ref = record.request.get("object_ref")
        target_ref = record.request.get("target_ref")
        if not isinstance(expected_value, Mapping) or not object_ref or not target_ref:
            return False, None
        try:
            snapshot = self.sdk.state.scene_snapshot()
        except (OSError, RuntimeError, ValueError):
            return False, None
        current = next(
            (item for item in snapshot.objects if item.source_id == str(object_ref)),
            None,
        )
        region = next(
            (item for item in snapshot.regions if item.source_id == str(target_ref)),
            None,
        )
        if current is None or region is None:
            return False, None
        expected = pose(expected_value)
        position_error_m = sqrt(
            sum(
                (float(current.pose.position[index]) - float(expected.position[index]))
                ** 2
                for index in range(3)
            )
        )
        within_target_xy = (
            sqrt(
                sum(
                    (
                        float(current.pose.position[index])
                        - float(region.pose.position[index])
                    )
                    ** 2
                    for index in (0, 1)
                )
            )
            <= self._TARGET_CENTER_FALLBACK_TOLERANCE_M
            if region.extent is None
            else all(
                abs(
                    float(current.pose.position[index])
                    - float(region.pose.position[index])
                )
                <= float(region.extent[index]) / 2
                for index in (0, 1)
            )
        )
        vertical_error_m = abs(
            float(current.pose.position[2]) - float(expected.position[2])
        )
        orientation_dot = abs(sum(
            float(left) * float(right)
            for left, right in zip(
                current.pose.quaternion_xyzw,
                expected.quaternion_xyzw,
                strict=True,
            )
        ))
        orientation_error_rad = 2.0 * acos(min(1.0, max(-1.0, orientation_dot)))
        external_contacts = self._external_object_contacts(str(object_ref))
        contact = bool(external_contacts)
        details = {
            "object_ref": str(object_ref),
            "target_ref": str(target_ref),
            "position_error_m": position_error_m,
            # 高度差只作为诊断信息保留。放置寻底时，规划位姿与真实接触面
            # 之间会包含模型、接触形变和托盘厚度误差；一旦箱体已进入目标
            # 区域并由外部支撑接触，就表示载荷已经从夹具转移。这里再叠加
            # 毫米级高度门限会把正常卸载误判为脱手。最终是否放稳、工具是否
            # 清空仍由 VerifyPlacement 独立验证，过载和传感器故障也不会被
            # 这条支撑转移规则吞掉。
            "vertical_error_m": vertical_error_m,
            "orientation_error_rad": orientation_error_rad,
            "orientation_tolerance_rad": MAX_SUPPORT_TRANSFER_ORIENTATION_ERROR_RAD,
            "within_target_xy": within_target_xy,
            "contact": contact,
            "external_contacts": external_contacts,
            "scene_generation": snapshot.generation,
        }
        return (
            within_target_xy
            and contact
            and orientation_error_rad <= MAX_SUPPORT_TRANSFER_ORIENTATION_ERROR_RAD,
            details,
        )

    def _external_object_contacts(self, object_ref: str) -> list[str]:
        """读取现有contact传感器，排除夹具自身接触后返回外部支撑身份。

        ``SceneObject.in_contact`` 同时包含工具接触，不能证明箱体已经落座。
        这里复用Runtime已有的通用公开接触对；目标区域和竖直几何仍由上方
        校验，因此不会把悬空夹持误当作支撑转移。
        """
        try:
            frame = self.sdk.sensors.latest("contact")
            payload = frame.payload
            if isinstance(payload, bytes):
                payload = json.loads(payload.decode("utf-8"))
        except (OSError, RuntimeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        if not isinstance(payload, Mapping):
            return []
        result: set[str] = set()
        for item in payload.get("contacts", []):
            if not isinstance(item, Mapping):
                continue
            first = str(item.get("first") or "")
            second = str(item.get("second") or "")
            if object_ref == first:
                other = second
            elif object_ref == second:
                other = first
            else:
                continue
            if other and not other.startswith(f"{self.sdk.robot_id}:"):
                result.add(other)
        return sorted(result)

    def execute(
        self, task_name: str, data: Mapping[str, Any], invocation_id: str, digest: str
    ) -> ExecutionRecord:
        if task_name == "LiftHeldObject":
            return self._lift(data, invocation_id, digest)
        if task_name == "MoveToPosture":
            return self._move_to_posture(data, invocation_id, digest)
        if task_name not in {"MoveEndEffector", "FollowWaypoints"}:
            raise ValueError(f"ManipulatorMotion 不支持 Task: {task_name}")
        target_values = data["targets"]
        targets = self._targets(target_values)
        purpose = data.get("purpose")
        required_end_effectors = {
            component_name(self.sdk, str(tool_ref), "manipulator_motion")
            for tool_ref in data.get("required_contact_tools", [])
        }
        fixed_end_effectors = required_end_effectors - set(targets)
        if purpose == "clearance" and data.get("expected_object_pose") is None:
            # clearance仍在箱体上方自由空间，只负责把第二只手从当前姿态带到
            # 紧凑接入起点。抓取阶段没有放置目标，不必把局部单臂动作升级
            # 成多末端IK。放置阶段会同时提交expected_object_pose；如果另一侧
            # 仍接触箱体，则必须保留实时固定末端约束，允许躯干与两臂同步
            # 补偿，但不能把已落座箱体拖离目标列。后续靠近
            # 凹槽的transfer/pregrasp/insert/seat仍保留固定承载侧约束。
            fixed_end_effectors = set()
        if fixed_end_effectors:
            # Skill只提交正在运动的第二侧目标。首侧固定Pose必须在Action开始时
            # 从Robot SDK实时读取；把旧Pose复制进Skill会在共享躯干运动后变成
            # 过期约束，并让夹具把箱体拉回旧位置。
            state = self.sdk.state.snapshot()
            missing = fixed_end_effectors - set(state.end_effectors)
            if missing:
                return self.failed(
                    task_name,
                    data,
                    invocation_id,
                    digest,
                    "END_EFFECTOR_POSE_UNAVAILABLE",
                    f"Robot SDK没有返回固定末端位姿：{sorted(missing)}",
                )
            targets.update(
                {name: state.end_effectors[name] for name in fixed_end_effectors}
            )
        # 只要Skill声明required_contact_tools，首侧就已经在凹槽内真实承载。
        # clearance、transfer和alignment都可能使用共享躯干；若alignment单独
        # 解除固定端投影，右臂接近的最后一段会把左下钩从槽内扭成侧向擦边。
        # 因此整段第二侧接近都保持Action开始时读取的首侧末端Pose。
        path_fixed_end_effectors = set(fixed_end_effectors)
        # pregrasp/insert/extract直接约束钩脚几何路径。抓取阶段的clearance
        # 仍使用右臂优先的连续关节轨迹；放置后运动侧已横向脱钩，其后续
        # clearance必须保持竖直离开，不能让关节捷径重新扫入箱体。
        # 首侧已经承载箱体时，第二侧的transfer发生在显式自由空间
        # 走廊中。此时若只做关节插值，末端会离开走廊并让钩脚扫过箱壁；
        # 因此保持固定首侧的同时约束第二侧笛卡尔路径。普通无载transfer
        # 仍使用连续关节轨迹，不把该约束扩散到所有运动。
        # place-object 打开夹具后会依次执行退钩、横撤和竖直离开，三段都
        # 使用 purpose=retreat。若退化为关节插值，双手端点虽已在箱外，
        # 中间扫掠仍可能让下钩重新进入凹槽并推偏已经落座的箱体。
        # 这里复用 SDK 已有笛卡尔路径，只约束真实退钩动作，不改变抓取
        # 候选生成或通用 IK。
        placement_clearance = (
            purpose == "clearance"
            and data.get("expected_object_pose") is not None
            and bool(fixed_end_effectors)
        )
        cartesian_motion = purpose in {
            "pregrasp", "insert", "extract", "unseat", "disengage", "retreat",
        } or placement_clearance
        contact_motion = purpose in {"insert", "seat"}
        # 预抓取阶段必须把目标箱体和邻箱都作为障碍；进入接触、外拉或携物
        # 阶段后，只排除本次明确操作的目标物体，邻箱仍参与整条轨迹检查。
        # 这里使用 Runtime SceneSnapshot，而不是 Semantic Map 中可能过期的位姿。
        exclude_object = purpose in {
            "insert",
            "seat",
            "extract",
            "transport",
            "place",
            # 释放后的 unseat/disengage 仍在槽内，允许目标箱体受控接触；
            # 横向退出完成后，retreat 必须恢复目标箱体碰撞，避免空钩穿过
            # 已放稳的箱体并把它推向Robot一侧。
            "unseat",
            "disengage",
        } or placement_clearance
        if placement_clearance:
            # 横向退钩已经由上一动作完成，Skill也只会在实时接触传感器确认
            # 运动侧为空后请求这段竖直上抬。周转箱的场景碰撞代理是完整
            # AABB，无法表达侧面凹槽；即使钩脚已离开真实箱体，它仍可能与
            # 代理在起点重叠并让规划在0秒被拒绝。这里只排除本次已落座对象，
            # 同时强制运动端沿笛卡尔方向离开；邻箱、托盘、自碰撞和固定承载
            # 端约束继续参与整条路径检查，不降低实际撤离高度。
            path_fixed_end_effectors = set(fixed_end_effectors)
        environment = scene_collision_environment(
            self.sdk,
            exclude_source_ids=(str(data["object_ref"]),)
            if exclude_object and data.get("object_ref")
            else (),
            allowed_contact_end_effectors_by_source={
                str(data["object_ref"]): tuple(sorted(required_end_effectors))
            }
            if required_end_effectors and not exclude_object and data.get("object_ref")
            else {},
        )
        try:
            command = self._move(
                targets,
                command_id=f"{invocation_id}:command",
                maximum_speed_mps=float(data["maximum_speed_mps"]),
                # hook_contact是Runtime提供的瞬时原始接触，既可能来自凹槽上沿，
                # 也可能来自凹槽背板。若在任意接触时结束插入，钩脚可能只停在
                # 槽口，随后闭合只能空行程。insert 因此完成短笛卡尔路径；Runtime
                # 仍按 contact_tool_refs/max_contact_force_n 在超力时立即失败并hold，
                # 是否正确钩入则由后续实时Observation判断。
                stop_on_contact=False,
                contact_tool_refs=[str(item["tool_ref"]) for item in target_values]
                if contact_motion
                else [],
                max_contact_force_n=(
                    data.get("max_contact_force_n") if contact_motion else None
                ),
                # 自由空间动作沿用RobotDeployment的通用容差；pregrasp是
                # insert的直接起点，二者之间只有约4cm短行程。若pregrasp在
                # 关节仍有通用9mrad残差时就返回，下一段多末端IK会因动态起点
                # 偶发不收敛。因此pregrasp与insert/seat都使用SDK已有的2mrad
                # 关节终点容差，再把稳定的当前状态交给下一段规划。
                # 这不是抓取成功判定：接触、受力和箱体跟随仍由后续实时观测
                # 独立确认，但不能让七轴误差叠加后把钩脚提前停在槽口外。
                # unseat/disengage仍可能位于凹槽附近，但要区分三种事实：
                # 有固定承载侧时，正在运动的工具仍参与持物约束，继续使用
                # 2mrad精密终点；两个工具已经由VerifyPlacement确认卸载后，
                # 空钩只需完成受碰撞检查的退出路径，layout001支撑面短推实测
                # 76mrad残差没有对应物理碰撞或载荷风险。此时使用80mrad，避免把
                # 99.9%完成的空工具撤离误报失败；实际逐采样碰撞检查不放松。
                # 支撑面短推是单工具、带最终箱体目标且不再要求夹具承重的
                # disengage。Runtime始终先执行完整轨迹，关节容差只判断轨迹
                # 结束后的稳定结果；箱体落座后接触阻力会留下约20～40mrad
                # 的关节残差，因此这里只使用50mrad，避免把已完成的真实推进
                # 记录为失败。动作后Skill仍以实时箱体位置、支撑和稳定状态
                # 判定是否完成，普通退钩、携物和自由空间动作均不受影响。
                position_tolerance_rad=(
                    0.002
                    if contact_motion or purpose == "pregrasp"
                    else 0.002
                    if purpose in {"unseat", "disengage"} and fixed_end_effectors
                    else 0.05
                    if purpose == "disengage"
                    and len(targets) == 1
                    and not required_end_effectors
                    and data.get("expected_object_pose") is not None
                    else 0.012
                    if purpose in {"unseat", "disengage"}
                    and (
                        required_end_effectors
                        or data.get("expected_object_pose") is not None
                    )
                    else 0.08
                    if purpose in {"unseat", "disengage"}
                    # place的几何目标会故意略穿过支撑面，用来表达“向下寻找真实
                    # 支撑接触”。因此不能用宽松的空间运动完成容差，否则箱底仍
                    # 悬空时Runtime就会返回成功。这里复用现有的1.2e-2 rad接触动作
                    # 容差；真实接触一旦成立，上方的支撑转移观测会立即停止命令。
                    # 这是运动执行容差，不是对箱体位置增加毫米级业务判定。
                    else 0.012 if purpose == "place"
                    else 0.012 if purpose == "retreat"
                    else None
                ),
                # 抓取阶段的clearance与transfer使用连续关节时间轨迹；放置后
                # 的clearance以及真实接触动作保留末端几何路径。SDK仍对整条
                # 轨迹逐采样验证固定首侧、自碰撞和邻箱碰撞。
                preserve_cartesian_path=cartesian_motion,
                # 只有两只工具已经通过同一物体形成物理耦合时，才保持
                # 整条轨迹中的相对几何。无载双手insert各自进入两侧凹槽，
                # 不应因此被重定时为十几秒的刚性双臂动作。
                preserve_relative_geometry=purpose in {"transport", "place"},
                environment=environment,
                fixed_end_effectors=path_fixed_end_effectors,
            )
        except PlanningError as exc:
            # IK 与碰撞规划完全发生在 Robot SDK 下发 Runtime 命令之前。这里
            # 登记确定失败，使 Pilot 可以继续选择新候选；网络或命令启动结果
            # 未知的错误仍由下层保留为 interrupted，不能在这里泛化吞掉。
            return self.failed(
                task_name, data, invocation_id, digest, "PLANNING_FAILED", str(exc)
            )
        return self.register_command(
            command,
            task_name,
            data,
            invocation_id,
            digest,
            resource="upper_body:" + "+".join(sorted(targets)),
            # Robot SDK 命令完成只说明关节轨迹已经进入底层容差，不能伪造成
            # “末端已满足语义预抓取条件”。末端与目标的真实误差由独立的
            # ObjectPerception.VerifyPregrasp 重新读取 Robot 状态后判断。
            result={"command_completed": True, "tools": sorted(targets)},
        )

    def _move_to_posture(
        self, data: Mapping[str, Any], invocation_id: str, digest: str
    ) -> ExecutionRecord:
        posture_name = str(data["posture"])
        postures = self.sdk.deployment.robot.kinematics.named_postures
        goal = postures.get(posture_name)
        if not goal:
            return self.failed(
                "MoveToPosture",
                data,
                invocation_id,
                digest,
                "POSTURE_NOT_CONFIGURED",
                f"RobotDeployment 未声明命名姿态：{posture_name}",
            )
        try:
            command = self.sdk.upper_body.move_joints(
                goal,
                command_id=f"{invocation_id}:command",
                environment=scene_collision_environment(self.sdk),
                speed_scale=1.0,
            )
        except PlanningError as exc:
            return self.failed(
                "MoveToPosture",
                data,
                invocation_id,
                digest,
                "PLANNING_FAILED",
                str(exc),
            )
        return self.register_command(
            command,
            "MoveToPosture",
            data,
            invocation_id,
            digest,
            resource="upper_body:posture",
            result={"posture": posture_name},
        )

    def _targets(self, values: list[Mapping[str, Any]]) -> dict[str, Pose]:
        return {
            component_name(self.sdk, str(item["tool_ref"]), "manipulator_motion"): pose(
                item["target_pose"]
            )
            for item in values
        }

    def _move(
        self,
        targets: dict[str, Pose],
        *,
        command_id: str,
        maximum_speed_mps: float,
        stop_on_contact: bool = False,
        contact_tool_refs: Sequence[str] = (),
        max_contact_force_n: float | None = None,
        position_tolerance_rad: float | None = None,
        preserve_cartesian_path: bool = False,
        preserve_relative_geometry: bool = False,
        environment=None,
        fixed_end_effectors=(),
    ):
        options = {
            "stop_on_contact": stop_on_contact,
            "contact_tool_refs": list(contact_tool_refs),
            "max_contact_force_n": max_contact_force_n,
            # 自由搬运沿用Deployment对真实执行器的通用容差；钩入和承力
            # 就位需要更精确地到达狭窄机械接口，由R1 Pro Ability Provider
            # 为本次动作收紧。该值不进入Robot Skill输入，也不表达仿真帧。
            "position_tolerance_rad": position_tolerance_rad,
        }
        if len(targets) == 1:
            name, target = next(iter(targets.items()))
            return self.sdk.upper_body.move_end_effector(
                name,
                target,
                command_id=command_id,
                maximum_speed_mps=maximum_speed_mps,
                preserve_cartesian_path=preserve_cartesian_path,
                environment=environment,
                **options,
            )
        return self.sdk.upper_body.move_end_effectors(
            targets,
            command_id=command_id,
            maximum_speed_mps=maximum_speed_mps,
            preserve_cartesian_path=preserve_cartesian_path,
            preserve_relative_geometry=preserve_relative_geometry,
            environment=environment,
            fixed_end_effectors=fixed_end_effectors,
            **options,
        )

    def _lift(
        self, data: Mapping[str, Any], invocation_id: str, digest: str
    ) -> ExecutionRecord:
        tool_refs = [str(ref) for ref in data["tools"]]
        state, load, _observed_ms = wait_for_confirmed_tool_engagement(
            self.sdk, tool_refs
        )
        # VerifyToolLoad 与下一条起升命令之间，接触传感器可能出现一个瞬时
        # 空采样。这里仅要求钩入、压紧和受力连续成立一个现有的短窗口，不再
        # 重复等待静态竖直承载；支撑方向和载荷转移中的相对运动仍由本 Action
        # 已有的连续监视负责。这样不会因单帧抖动拒绝正常抓取，也不会让持续
        # 缺钩或传感故障进入起升。
        deferred_reasons = (":vertical_support_low", ":relative_motion_high")
        blocking_reasons = [
            reason
            for reason in load.reasons
            if not reason.endswith(deferred_reasons)
        ]
        attachment_intact = bool(load.tool_states) and not (
            blocking_reasons or load.overloaded or load.sensor_fault
        )
        if not attachment_intact:
            return self.failed(
                "LiftHeldObject",
                data,
                invocation_id,
                digest,
                "LOAD_NOT_STABLE",
                "双侧夹具接触、受力或传感状态不完整，禁止开始抬升: "
                + ", ".join(blocking_reasons),
            )
        current_poses: dict[str, Pose] = {}
        for tool_ref in data["tools"]:
            end_effector = component_name(self.sdk, str(tool_ref), "manipulator_motion")
            current = state.end_effectors.get(end_effector)
            if current is None:
                return self.failed(
                    "LiftHeldObject",
                    data,
                    invocation_id,
                    digest,
                    "END_EFFECTOR_POSE_UNAVAILABLE",
                    f"Robot SDK 没有返回 {end_effector} 末端位姿",
                )
            current_poses[end_effector] = current

        def lift_targets(distance_m: float) -> dict[str, Pose]:
            return {
                end_effector: current.model_copy(
                    update={
                        "position": (
                            current.position[0],
                            current.position[1],
                            current.position[2] + distance_m,
                        )
                    }
                )
                for end_effector, current in current_poses.items()
            }

        requested_distance_m = float(data["distance_m"])
        effective_data: Mapping[str, Any] = data
        targets = lift_targets(requested_distance_m)
        try:
            command = self._move(
                targets,
                command_id=f"{invocation_id}:command",
                maximum_speed_mps=float(data["maximum_speed_mps"]),
                # 只有已经稳定承载物体的运动才必须在整条路径上保持双末端
                # 相对几何；无载接近只需同步到达终点，否则会产生大量无意义
                # 路点并显著拉长动作时间。
                preserve_cartesian_path=True,
                preserve_relative_geometry=True,
                environment=scene_collision_environment(
                    self.sdk,
                    exclude_source_ids=(str(data["object_ref"]),),
                ),
            )
        except PlanningError as exc:
            fallback_distance_m = requested_distance_m * 0.5
            if fallback_distance_m < 0.01:
                return self.failed(
                    "LiftHeldObject",
                    data,
                    invocation_id,
                    digest,
                    "PLANNING_FAILED",
                    str(exc),
                )
            # layout001 的双侧工具刚完成接合时，当前姿态可能处在可达域边缘。
            # 首次规划失败尚未向 Runtime 下发任何命令，因此允许把本段距离
            # 减半后重新规划一次；这不是重放物理动作，也不修改通用 IK。
            # 实际完成的距离写回 Action 结果，Robot Skill仍用实时物体高度
            # 决定是否继续，不能把原请求距离冒充已经完成的物理进度。
            effective_data = {**data, "distance_m": fallback_distance_m}
            targets = lift_targets(fallback_distance_m)
            try:
                command = self._move(
                    targets,
                    command_id=f"{invocation_id}:command",
                    maximum_speed_mps=float(data["maximum_speed_mps"]),
                    preserve_cartesian_path=True,
                    preserve_relative_geometry=True,
                    environment=scene_collision_environment(
                        self.sdk,
                        exclude_source_ids=(str(data["object_ref"]),),
                    ),
                )
            except PlanningError as fallback_exc:
                return self.failed(
                    "LiftHeldObject",
                    data,
                    invocation_id,
                    digest,
                    "PLANNING_FAILED",
                    str(fallback_exc),
                )
        return self.register_command(
            command,
            "LiftHeldObject",
            effective_data,
            invocation_id,
            digest,
            resource="upper_body:" + "+".join(sorted(targets)),
            result={
                "lift_height_m": effective_data["distance_m"],
                "tools": list(data["tools"]),
            },
        )
