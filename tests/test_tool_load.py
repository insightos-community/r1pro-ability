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

"""基于通用实时传感状态的工具承载判断测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from semantic_robot_sdk_core import RobotState, ToolDescriptor, ToolState

from r1pro_abilities.tool_load import (
    assess_tool_load,
    ToolLoadAssessment,
    ToolLoadPolicy,
    tools_are_unloaded,
    wait_for_stable_tool_load,
)


TOOL_REFS = ("component://tool/left", "component://tool/right")


def _assessment(*, safe: bool, engaged: bool | None = None) -> ToolLoadAssessment:
    return ToolLoadAssessment(
        engaged=safe if engaged is None else engaged,
        slipping=not safe,
        overloaded=False,
        sensor_fault=False,
        reasons=() if safe else ("component://tool/left:relative_motion_high",),
        tool_states={ref: None for ref in TOOL_REFS},
    )


class ToolLoadTest(unittest.TestCase):
    def test_normal_carrying_compliance_is_not_reported_as_slipping(self) -> None:
        """接触与承力正常时，厘米级相对摆动不是箱体脱手。"""

        descriptors = [
            ToolDescriptor(
                tool_ref=ref, side=side, kind="tote_clamp", frame=f"{side}_tool",
                joint=f"{side}_tool_joint", travel_m=0.04,
                normal_force_n=60.0, maximum_force_n=120.0,
            )
            for ref, side in zip(TOOL_REFS, ("left", "right"), strict=True)
        ]
        state = RobotState(
            robot_id="r1pro-test", generation=1, joint_positions={},
            tool_states={
                ref: ToolState(
                    tool_ref=ref, side=side, kind="tote_clamp",
                    position=0.012, velocity=0.0, effort=40.0,
                    hook_contact=True, hook_force_n=40.0,
                    hook_support_ratio=0.99, clamp_contact=True,
                    clamp_force_n=30.0, hook_tangential_speed_m_s=0.03,
                )
                for ref, side in zip(TOOL_REFS, ("left", "right"), strict=True)
            },
        )

        assessment = assess_tool_load(state, descriptors, TOOL_REFS)

        self.assertTrue(assessment.safe)
        self.assertNotIn("relative_motion_high", " ".join(assessment.reasons))

    def test_unloaded_tool_may_keep_non_supporting_hook_contact(self) -> None:
        """横向擦碰有接触和微小力，但不构成工具继续承载。"""

        descriptor = ToolDescriptor(
            tool_ref=TOOL_REFS[0], side="left", kind="tote_clamp",
            frame="left_tool", joint="left_tool_joint", travel_m=0.04,
            normal_force_n=60.0, maximum_force_n=120.0,
        )
        state = RobotState(
            robot_id="r1pro-test", generation=1, joint_positions={},
            tool_states={
                TOOL_REFS[0]: ToolState(
                    tool_ref=TOOL_REFS[0], side="left", kind="tote_clamp",
                    position=0.035, velocity=0.0, effort=1.0,
                    hook_contact=True, hook_force_n=5.6,
                    hook_support_ratio=0.0004, clamp_contact=False,
                )
            },
        )

        self.assertTrue(tools_are_unloaded(state, [descriptor], [TOOL_REFS[0]]))

    def test_transient_relative_motion_restarts_stability_window(self) -> None:
        """短时振动不抹掉接触事实，但连续安全时间必须重新累计。"""

        state = object()
        policy = ToolLoadPolicy(
            stable_duration_s=0.12,
            acquisition_timeout_s=0.60,
            sample_interval_s=0.02,
        )
        with (
            patch(
                "r1pro_abilities.tool_load.monotonic",
                side_effect=[0.00, 0.01, 0.07, 0.14, 0.27],
            ),
            patch("r1pro_abilities.tool_load.sleep"),
            patch(
                "r1pro_abilities.tool_load.read_tool_load",
                side_effect=[
                    (state, _assessment(safe=True)),
                    (state, _assessment(safe=False, engaged=True)),
                    (state, _assessment(safe=True)),
                    (state, _assessment(safe=True)),
                ],
            ),
        ):
            _, assessment, observed_ms = wait_for_stable_tool_load(
                object(), TOOL_REFS, policy
            )

        self.assertTrue(assessment.safe)
        self.assertEqual(observed_ms, 130)

    def test_transient_relative_motion_can_settle_within_acquisition_window(self) -> None:
        """闭合后的一个高速采样不应覆盖随后持续稳定的真实状态。"""

        state = object()
        policy = ToolLoadPolicy(
            stable_duration_s=0.12,
            acquisition_timeout_s=0.60,
            sample_interval_s=0.02,
        )
        with (
            patch(
                "r1pro_abilities.tool_load.monotonic",
                side_effect=[0.00, 0.01, 0.05, 0.18],
            ),
            patch("r1pro_abilities.tool_load.sleep"),
            patch(
                "r1pro_abilities.tool_load.read_tool_load",
                side_effect=[
                    (state, _assessment(safe=False)),
                    (state, _assessment(safe=True)),
                    (state, _assessment(safe=True)),
                ],
            ),
        ):
            returned_state, assessment, observed_ms = wait_for_stable_tool_load(
                object(), TOOL_REFS, policy
            )

        self.assertIs(returned_state, state)
        self.assertTrue(assessment.safe)
        self.assertEqual(observed_ms, 130)


    def test_persistent_relative_motion_fails_at_acquisition_deadline(self) -> None:
        """相对运动始终不收敛时必须在有限窗口失败，不能无限等待或放宽条件。"""

        state = object()
        policy = ToolLoadPolicy(
            stable_duration_s=0.12,
            acquisition_timeout_s=0.60,
            sample_interval_s=0.02,
        )
        with (
            patch(
                "r1pro_abilities.tool_load.monotonic",
                side_effect=[0.00, 0.01, 0.61],
            ),
            patch("r1pro_abilities.tool_load.sleep"),
            patch(
                "r1pro_abilities.tool_load.read_tool_load",
                side_effect=[
                    (state, _assessment(safe=False)),
                    (state, _assessment(safe=False)),
                ],
            ),
        ):
            returned_state, assessment, observed_ms = wait_for_stable_tool_load(
                object(), TOOL_REFS, policy
            )

        self.assertIs(returned_state, state)
        self.assertTrue(assessment.slipping)
        self.assertEqual(observed_ms, 0)
