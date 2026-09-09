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

"""七类 R1 Pro Ability 的公共调用入口。

公共层只处理严格输入、Robot 归属、幂等 invocation 和执行恢复。各 Ability
自己的业务判断放在独立 Handler 中，避免再次形成一个包含整台机器人的大分派器。
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from pydantic import BaseModel
from semantic_robot_sdk_r1pro import R1ProSDK

from .ability_utils import invocation_id as make_invocation_id
from .ability_utils import request_digest
from .catalog import TASKS, AbilityRole
from .execution import ExecutionRegistry
from .handlers import (
    EndEffectorHandler,
    GraspPlanningHandler,
    ManipulatorMotionHandler,
    NavigationHandler,
    ObjectPerceptionHandler,
    RobotStateHandler,
    SensorCaptureHandler,
)
from .models import ModelProfileRegistry
from .task_models import TASK_INPUTS


_HANDLERS = {
    AbilityRole.NAVIGATION: NavigationHandler,
    AbilityRole.MANIPULATOR_MOTION: ManipulatorMotionHandler,
    AbilityRole.END_EFFECTOR: EndEffectorHandler,
    AbilityRole.ROBOT_STATE: RobotStateHandler,
    AbilityRole.SENSOR_CAPTURE: SensorCaptureHandler,
    AbilityRole.OBJECT_PERCEPTION: ObjectPerceptionHandler,
    AbilityRole.GRASP_PLANNING: GraspPlanningHandler,
}


class R1ProAbilityService:
    """一个实例只暴露一个 Ability 角色及其 Task。"""

    def __init__(
        self,
        role: AbilityRole,
        sdk: R1ProSDK,
        models: ModelProfileRegistry | None = None,
        execution_store_path: str | Path = ":memory:",
        artifact_exchange_root: str | Path | None = None,
    ) -> None:
        self.role = role
        self.sdk = sdk
        self.executions = ExecutionRegistry(execution_store_path)
        self.handler = _HANDLERS[role](
            sdk,
            self.executions,
            models,
            artifact_exchange_root,
        )
        self._start_lock = RLock()

    def close(self) -> None:
        self.executions.close()

    def invoke(self, task_name: str, input_data: Mapping[str, Any]) -> dict[str, Any]:
        model_type = TASK_INPUTS.get(task_name)
        if model_type is None:
            raise ValueError(f"未知 Task: {task_name}")
        request = model_type.model_validate(dict(input_data))
        if task_name == "GetExecution":
            record = self.executions.refresh(self.sdk, request.invocation_id)
            record = self.handler.after_refresh(record)
            return record.to_dict(request.after_feedback_sequence)
        if task_name == "StopExecution":
            return self.executions.stop(
                self.sdk, request.invocation_id, request.reason
            ).to_dict()
        if task_name not in TASKS[self.role]:
            raise ValueError(f"{self.role.value} 不支持 Task: {task_name}")
        self._verify_robot(request)
        return self._invoke_business_task(task_name, request)

    def _verify_robot(self, request: BaseModel) -> None:
        """Pilot 已精确路由；Ability 再拒绝一次跨 Robot 调用。"""

        robot_id = getattr(request, "robot_id", None)
        if robot_id is not None and robot_id != self.sdk.robot_id:
            raise ValueError(
                f"Ability 绑定 Robot {self.sdk.robot_id}，拒绝请求中的 {robot_id}"
            )

    def _invoke_business_task(
        self, task_name: str, request: BaseModel
    ) -> dict[str, Any]:
        data = request.model_dump(mode="json")
        supplied_invocation = data.pop("invocation_id", None)
        digest = request_digest({"task": task_name, "input": data})
        invocation = supplied_invocation or make_invocation_id(task_name, digest)
        with self._start_lock:
            existing = self.executions.find(invocation)
            if existing is not None:
                if existing.task_name != task_name or existing.request_digest != digest:
                    raise ValueError(f"invocation_id {invocation} 已用于不同内容")
                record = self.executions.refresh(self.sdk, invocation)
                return self.handler.after_refresh(record).to_dict()
            record = self.handler.execute(task_name, data, invocation, digest)
            return record.to_dict()


__all__ = [
    "TASKS",
    "AbilityRole",
    "R1ProAbilityService",
]
