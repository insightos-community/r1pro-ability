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

"""Ability 层基于实时工具传感器判断承载状态。

Runtime 与 Robot SDK 只提供当前接触、力、方向分量和相对速度。是否已经
持续满足某个动作的安全条件属于 Ability 语义，不能由仿真物理帧数或真机
驱动采样计数决定。这里用单调时钟观察一段真实时间，使 MuJoCo 与真机遵循
相同边界；Robot Skill 只消费最终 Observation，不理解这些传感器阈值。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic, sleep
from typing import Any, Callable, Sequence

from semantic_robot_sdk_core import RobotState, ToolDescriptor, ToolState
from semantic_robot_sdk_r1pro import R1ProSDK


@dataclass(frozen=True)
class ToolLoadPolicy:
    """夹具传感判断策略；阈值描述工具安全性，不描述箱体尺寸或场景。"""

    minimum_hook_force_n: float = 1.0
    minimum_clamp_force_n: float = 2.0
    minimum_support_ratio: float = 0.60
    # MuJoCo 携物转向时，钩脚沿箱沿会稳定出现约 0.03 m/s 的相对运动；
    # 四个接触、承力方向和受力都正常时，这属于夹具顺应性而不是脱手。
    # 使用厘米级 0.05 m/s 边界，仍由接触丢失、支撑下降、超力和传感
    # 故障独立触发停止，也会保留更高相对速度作为真实滑移信号。
    maximum_tangential_speed_m_s: float = 0.05
    maximum_force_ratio: float = 1.0
    stable_duration_s: float = 0.12
    acquisition_timeout_s: float = 1.50
    sample_interval_s: float = 0.02


DEFAULT_TOOL_LOAD_POLICY = ToolLoadPolicy()


@dataclass(frozen=True)
class ToolLoadAssessment:
    engaged: bool
    slipping: bool
    overloaded: bool
    sensor_fault: bool
    reasons: tuple[str, ...]
    tool_states: dict[str, ToolState | None]

    @property
    def safe(self) -> bool:
        return self.engaged and not (
            self.slipping or self.overloaded or self.sensor_fault
        )

    def value(self, *, observed_duration_ms: int = 0) -> dict[str, Any]:
        return {
            "stable_bilateral_load": self.safe,
            "slip_detected": self.slipping,
            "overload_detected": self.overloaded,
            "sensor_fault": self.sensor_fault,
            "observed_duration_ms": observed_duration_ms,
            "reasons": list(self.reasons),
            "tool_states": {
                ref: state.model_dump(mode="json") if state is not None else None
                for ref, state in self.tool_states.items()
            },
        }


def assess_tool_load(
    state: RobotState,
    descriptors: Sequence[ToolDescriptor],
    tool_refs: Sequence[str],
    policy: ToolLoadPolicy = DEFAULT_TOOL_LOAD_POLICY,
) -> ToolLoadAssessment:
    """根据一次实时采样计算当前状态，不持久化、不累计 Runtime 帧。"""

    descriptor_by_ref = {item.tool_ref: item for item in descriptors}
    selected = {ref: state.tool_states.get(ref) for ref in tool_refs}
    reasons: list[str] = []
    engagement_reasons: list[str] = []
    slipping = False
    overloaded = False
    sensor_fault = False

    for ref, value in selected.items():
        descriptor = descriptor_by_ref.get(ref)
        if descriptor is None:
            reason = f"{ref}:descriptor_unavailable"
            reasons.append(reason)
            engagement_reasons.append(reason)
            sensor_fault = True
            continue
        if value is None:
            reason = f"{ref}:state_unavailable"
            reasons.append(reason)
            engagement_reasons.append(reason)
            sensor_fault = True
            continue
        if value.sensor_fault:
            reason = f"{ref}:sensor_fault"
            reasons.append(reason)
            engagement_reasons.append(reason)
            sensor_fault = True
        if not value.hook_contact:
            reason = f"{ref}:hook_contact_missing"
            reasons.append(reason)
            engagement_reasons.append(reason)
        if not value.clamp_contact:
            reason = f"{ref}:clamp_contact_missing"
            reasons.append(reason)
            engagement_reasons.append(reason)
        if value.hook_force_n < policy.minimum_hook_force_n:
            reason = f"{ref}:hook_force_low"
            reasons.append(reason)
            engagement_reasons.append(reason)
        if value.clamp_force_n < policy.minimum_clamp_force_n:
            reason = f"{ref}:clamp_force_low"
            reasons.append(reason)
            engagement_reasons.append(reason)
        if value.hook_support_ratio < policy.minimum_support_ratio:
            reason = f"{ref}:vertical_support_low"
            reasons.append(reason)
            engagement_reasons.append(reason)
        if value.hook_tangential_speed_m_s > policy.maximum_tangential_speed_m_s:
            reasons.append(f"{ref}:relative_motion_high")
            slipping = True
        force_limit = descriptor.maximum_force_n * policy.maximum_force_ratio
        if max(value.hook_force_n, value.clamp_force_n) > force_limit:
            reasons.append(f"{ref}:force_limit_exceeded")
            overloaded = True

    return ToolLoadAssessment(
        # “已经钩入并压紧”与“当前存在相对运动”是两件不同的事实。
        # 如果把瞬时速度也塞进 engaged，机械振动会让整个接触稳定窗口反复
        # 归零，最终在两侧接触和受力都成立时仍错误拒绝抬升。
        engaged=bool(selected) and not engagement_reasons,
        slipping=slipping,
        overloaded=overloaded,
        sensor_fault=sensor_fault,
        reasons=tuple(reasons),
        tool_states=selected,
    )


def tools_are_unloaded(
    state: RobotState,
    descriptors: Sequence[ToolDescriptor],
    tool_refs: Sequence[str],
) -> bool:
    """判断指定工具均未形成承载，同时拒绝未知、超力或传感故障状态。"""

    if not tool_refs:
        return False
    for ref in tool_refs:
        load = assess_tool_load(state, descriptors, [ref])
        if load.engaged or load.overloaded or load.sensor_fault:
            return False
    return True


def read_tool_load(
    sdk: R1ProSDK,
    tool_refs: Sequence[str],
    policy: ToolLoadPolicy = DEFAULT_TOOL_LOAD_POLICY,
) -> tuple[RobotState, ToolLoadAssessment]:
    state = sdk.state.snapshot()
    return state, assess_tool_load(
        state, sdk.state.capabilities().tools, tool_refs, policy
    )


def parse_load_unstable_since(value: object) -> datetime | None:
    """解析 Ability Execution 中保存的真实时间起点。"""

    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def load_failure_requires_stop(
    load: ToolLoadAssessment,
    unstable_since: datetime,
    policy: ToolLoadPolicy = DEFAULT_TOOL_LOAD_POLICY,
) -> bool:
    """判断当前原始传感异常是否已达到 Action 停止边界。"""

    if load.sensor_fault or load.overloaded:
        return True
    return (datetime.now(timezone.utc) - unstable_since).total_seconds() >= policy.stable_duration_s


def _wait_for_tool_load(
    sdk: R1ProSDK,
    tool_refs: Sequence[str],
    accepted: Callable[[ToolLoadAssessment], bool],
    policy: ToolLoadPolicy = DEFAULT_TOOL_LOAD_POLICY,
) -> tuple[RobotState, ToolLoadAssessment, int]:
    """按真实时间连续观察工具状态，不依赖 Runtime 帧数。"""
    deadline = monotonic() + policy.acquisition_timeout_s
    stable_since: float | None = None
    last_state: RobotState | None = None
    last_assessment: ToolLoadAssessment | None = None
    while True:
        now = monotonic()
        last_state, last_assessment = read_tool_load(sdk, tool_refs, policy)
        if last_assessment.sensor_fault or last_assessment.overloaded:
            return last_state, last_assessment, 0
        if accepted(last_assessment):
            stable_since = stable_since or now
        else:
            stable_since = None
        if (
            stable_since is not None
            and now - stable_since >= policy.stable_duration_s
        ):
            observed_ms = round((now - stable_since) * 1000)
            return last_state, last_assessment, observed_ms
        if now >= deadline:
            return last_state, last_assessment, 0
        sleep(policy.sample_interval_s)


def wait_for_confirmed_tool_engagement(
    sdk: R1ProSDK,
    tool_refs: Sequence[str],
    policy: ToolLoadPolicy = DEFAULT_TOOL_LOAD_POLICY,
) -> tuple[RobotState, ToolLoadAssessment, int]:
    """等待双侧钩入和压紧持续成立，但不把取载前的相对运动当成脱手。

    CloseUntilContact 的职责只到“工具已经可靠接触物体”。箱体从托盘向夹具
    转移载荷时必然会有短暂相对运动；是否在抬升中失稳由运动 Ability 持续
    观察，最终是否形成稳定承载由 VerifyGrasp 独立确认。这样既保留原始速度
    供诊断，也避免三层重复使用同一个经验阈值阻止动作继续。
    """
    deferred_reasons = (":vertical_support_low", ":relative_motion_high")

    def attachment_intact(load: ToolLoadAssessment) -> bool:
        blocking_reasons = [
            reason
            for reason in load.reasons
            if not reason.endswith(deferred_reasons)
        ]
        return bool(load.tool_states) and not (
            blocking_reasons or load.overloaded or load.sensor_fault
        )

    return _wait_for_tool_load(sdk, tool_refs, attachment_intact, policy)


def wait_for_stable_tool_load(
    sdk: R1ProSDK,
    tool_refs: Sequence[str],
    policy: ToolLoadPolicy = DEFAULT_TOOL_LOAD_POLICY,
) -> tuple[RobotState, ToolLoadAssessment, int]:
    """等待连续安全承载，供起升前复核和最终持物验证使用。"""
    return _wait_for_tool_load(sdk, tool_refs, lambda load: load.safe, policy)
