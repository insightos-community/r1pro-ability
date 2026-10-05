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

"""Ability Execution 持久化、反馈游标和协作停止。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any, Mapping
from uuid import uuid4

from semantic_robot_sdk_core import (
    BackendUnavailable,
    Command,
    CommandState,
    RobotSDKError,
)
from semantic_robot_sdk_r1pro import R1ProSDK


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionStatus(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    STOPPING = "stopping"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"


@dataclass
class ExecutionRecord:
    invocation_id: str
    task_name: str
    request_digest: str
    operation: str
    command_id: str
    resource: str
    status: ExecutionStatus
    progress: float
    phase: str
    command_ids: tuple[str, ...] = ()
    request: Mapping[str, Any] = field(default_factory=dict)
    feedback_cursor: int = 0
    feedback: tuple[Mapping[str, Any], ...] = ()
    observations: tuple[Mapping[str, Any], ...] = ()
    result: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None
    updated_at: datetime = field(default_factory=_now)

    def to_dict(self, after_sequence: int = 0) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "task_name": self.task_name,
            "operation": self.operation,
            "command_id": self.command_id,
            "command_ids": list(self.command_ids),
            "resource": self.resource,
            "status": self.status.value,
            "progress": self.progress,
            "phase": self.phase,
            "feedback_cursor": self.feedback_cursor,
            "feedback": [
                dict(item)
                for item in self.feedback
                if int(item["sequence"]) > after_sequence
            ],
            "observations": [dict(item) for item in self.observations],
            "result": {
                key: value
                for key, value in self.result.items()
                if not key.startswith("_")
            },
            "error": dict(self.error) if self.error else None,
            "updated_at": self.updated_at.isoformat(),
        }


class ExecutionRegistry:
    """保存 invocation 与 SDK command 的对应关系，重启后只查询、不重放。"""

    def __init__(self, database_path: str | Path = ":memory:") -> None:
        self._records: dict[str, ExecutionRecord] = {}
        self._lock = RLock()
        self._db = sqlite3.connect(str(database_path), check_same_thread=False)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS ability_executions (
                invocation_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self._load()

    def close(self) -> None:
        self._db.close()

    def register(
        self,
        command: Command,
        request_digest: str,
        invocation_id: str,
        task_name: str,
        *,
        operation: str,
        resource: str,
        request: Mapping[str, Any],
        result_template: Mapping[str, Any] | None = None,
        observations: tuple[Mapping[str, Any], ...] = (),
    ) -> ExecutionRecord:
        record = ExecutionRecord(
            invocation_id=invocation_id,
            task_name=task_name,
            request_digest=request_digest,
            operation=operation,
            command_id=command.command_id,
            resource=resource,
            status=_execution_status(command.status),
            progress=command.progress,
            phase=command.status.value,
            command_ids=(command.command_id,),
            request=dict(request),
            observations=observations,
            # 异步命令登记时只能保存“成功后应返回什么”，不能提前公开成功结果。
            # 否则底层命令稍后失败时，Pilot 会同时收到 failed 和
            # command_completed=true，进而让 Robot Skill 误判动作已经完成。
            result={"_success_result": dict(result_template or {})},
        )
        self._publish_success_result(record)
        with self._lock:
            if invocation_id in self._records:
                raise ValueError(f"invocation_id 已存在: {invocation_id}")
            self._records[invocation_id] = record
            self._save(record)
        return record

    def register_many(
        self,
        commands: tuple[Command, ...],
        request_digest: str,
        invocation_id: str,
        task_name: str,
        *,
        operation: str,
        resource: str,
        request: Mapping[str, Any],
        result_template: Mapping[str, Any] | None = None,
    ) -> ExecutionRecord:
        """把一个双工具语义动作登记为单一 Ability Execution。

        MuJoCo 按工具分别返回 command ID。两条命令必须一起持久化、查询和停止，
        否则 Pilot 可能在一侧仍运动时把整个动作误报为完成。
        """

        if not commands:
            raise ValueError("复合执行至少包含一个底层命令")
        states = tuple(command.status for command in commands)
        status = _combined_execution_status(states)
        record = ExecutionRecord(
            invocation_id=invocation_id,
            task_name=task_name,
            request_digest=request_digest,
            operation=operation,
            command_id=commands[0].command_id,
            resource=resource,
            status=status,
            progress=sum(command.progress for command in commands) / len(commands),
            phase=status.value,
            command_ids=tuple(command.command_id for command in commands),
            request=dict(request),
            result={
                "_success_result": dict(result_template or {}),
                "_feedback_cursors": {},
            },
        )
        self._publish_success_result(record)
        with self._lock:
            if invocation_id in self._records:
                raise ValueError(f"invocation_id 已存在: {invocation_id}")
            self._records[invocation_id] = record
            self._save(record)
        return record

    def find(self, invocation_id: str) -> ExecutionRecord | None:
        with self._lock:
            return self._records.get(invocation_id)

    def find_result_value(self, key: str, value: Any) -> ExecutionRecord | None:
        with self._lock:
            return next(
                (
                    record
                    for record in self._records.values()
                    if record.result.get(key) == value
                ),
                None,
            )

    def register_succeeded(
        self,
        task_name: str,
        request_digest: str,
        result: Mapping[str, Any],
        invocation_id: str | None = None,
        observations: tuple[Mapping[str, Any], ...] = (),
        request: Mapping[str, Any] | None = None,
    ) -> ExecutionRecord:
        invocation = invocation_id or str(uuid4())
        record = ExecutionRecord(
            invocation_id=invocation,
            task_name=task_name,
            request_digest=request_digest,
            operation=task_name,
            command_id="",
            resource="compute_or_observation",
            status=ExecutionStatus.SUCCEEDED,
            progress=1.0,
            phase="completed",
            request=dict(request or {}),
            observations=observations,
            result=dict(result),
        )
        with self._lock:
            if invocation in self._records:
                raise ValueError(f"invocation_id 已存在: {invocation}")
            self._records[invocation] = record
            self._save(record)
        return record

    def register_terminal(
        self,
        task_name: str,
        request_digest: str,
        status: ExecutionStatus,
        result: Mapping[str, Any],
        invocation_id: str | None = None,
        observations: tuple[Mapping[str, Any], ...] = (),
        error: Mapping[str, Any] | None = None,
        request: Mapping[str, Any] | None = None,
    ) -> ExecutionRecord:
        if status not in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.STOPPED,
            ExecutionStatus.INTERRUPTED,
        }:
            raise ValueError("同步结果必须是终态")
        record = self.register_succeeded(
            task_name,
            request_digest,
            result,
            invocation_id,
            observations,
            request,
        )
        record.status = status
        record.phase = status.value
        record.error = dict(error) if error else None
        with self._lock:
            self._save(record)
        return record

    def update_observations(
        self,
        invocation_id: str,
        observations: tuple[Mapping[str, Any], ...],
        result: Mapping[str, Any] | None = None,
        *,
        status: ExecutionStatus | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> ExecutionRecord:
        with self._lock:
            record = self._require(invocation_id)
            record.observations = observations
            if result is not None:
                record.result = {**record.result, **dict(result)}
            if status is not None:
                if status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.STOPPED,
                    ExecutionStatus.INTERRUPTED,
                }:
                    raise ValueError("Observation 只能收敛到终态")
                if status is not ExecutionStatus.SUCCEEDED:
                    success_result = record.result.get("_success_result")
                    if isinstance(success_result, Mapping):
                        for key in success_result:
                            record.result.pop(str(key), None)
                record.status = status
                record.phase = status.value
                record.error = dict(error) if error else None
            record.updated_at = _now()
            self._save(record)
            return record

    def refresh(self, sdk: R1ProSDK, invocation_id: str) -> ExecutionRecord:
        with self._lock:
            record = self._require(invocation_id)
            terminal_before_refresh = record.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.STOPPED,
                ExecutionStatus.INTERRUPTED,
            }
            # Fake、部分真机固件会在启动调用返回前完成短命令。此时命令终态
            # 已经可信，但 SDK 的按序 Feedback 仍需读取一次并交给 Pilot。
            # 已读取过反馈或没有底层命令的同步 Task 可直接返回，避免无意义查询。
            if terminal_before_refresh and (
                not record.command_id or record.feedback_cursor > 0
            ):
                return record
            if not record.command_id:
                return record
            command_ids = record.command_ids or (record.command_id,)
            if len(command_ids) > 1:
                return self._refresh_many_locked(sdk, record, command_ids)
            try:
                feedback = list(sdk.commands.feedback(record.command_id))
                command = sdk.commands.get(record.command_id)
            except BackendUnavailable:
                # 一次状态读取超时只说明“这次没有读到”，不能证明物理命令已经
                # 丢失。把它直接收敛为 interrupted 会留下仍在运动的 Robot，
                # 也会诱使上层把可恢复的网络抖动当成不可恢复事故。保持原终态
                # 之前的状态，下一次 GetExecution 继续按同一 command_id 对账；
                # Pilot 的有界执行超时仍负责真正长期失联时的 stop/hold。
                record.phase = "command_status_temporarily_unavailable"
                record.updated_at = _now()
                self._save(record)
                return record
            except (KeyError, RobotSDKError) as error:
                # 已确认的终态不能因稍后的设备历史清理被降级为 interrupted；
                # 反馈缺失只影响进度证据，不改变已经取得的命令终态。
                if terminal_before_refresh:
                    return record
                record.status = ExecutionStatus.INTERRUPTED
                record.phase = "command_unknown_after_restart"
                record.error = {
                    "code": "COMMAND_UNKNOWN",
                    "message": f"底层命令无法确认，禁止重放: {error}",
                }
                record.updated_at = _now()
                self._save(record)
                return record

            appended = tuple(
                {
                    "sequence": item.sequence,
                    "status": command.status.value,
                    "phase": command.status.value,
                    "progress": item.progress,
                    "message": item.message
                    or f"{record.task_name}: {command.status.value}",
                    "severity": "info",
                    "measurements": {},
                    "observations": [],
                    "evidence_refs": [],
                    "timestamp": item.observed_at.isoformat(),
                }
                for item in feedback
                if item.sequence > record.feedback_cursor
            )
            if appended:
                record.feedback = (*record.feedback, *appended)
                record.feedback_cursor = int(appended[-1]["sequence"])
            record.progress = command.progress
            record.phase = command.status.value
            record.status = _execution_status(command.status)
            self._publish_success_result(record)
            if record.status is ExecutionStatus.FAILED:
                record.error = {
                    "code": "ROBOT_COMMAND_FAILED",
                    "message": command.reason or "Robot SDK 命令失败",
                }
            elif record.status is ExecutionStatus.INTERRUPTED:
                record.error = {
                    "code": "ROBOT_COMMAND_INTERRUPTED",
                    "message": command.reason or "Robot SDK 命令状态无法确认",
                }
            record.updated_at = _now()
            self._save(record)
            return record

    def _refresh_many_locked(
        self,
        sdk: R1ProSDK,
        record: ExecutionRecord,
        command_ids: tuple[str, ...],
    ) -> ExecutionRecord:
        try:
            commands = tuple(sdk.commands.get(value) for value in command_ids)
            feedback = {
                value: tuple(sdk.commands.feedback(value)) for value in command_ids
            }
        except BackendUnavailable:
            # 复合命令与单命令使用相同规则：任何一侧暂时查不到都保留整组
            # Execution，不能把另一侧仍在执行的动作误报为 interrupted。
            record.phase = "compound_status_temporarily_unavailable"
            record.updated_at = _now()
            self._save(record)
            return record
        except (KeyError, RobotSDKError) as error:
            record.status = ExecutionStatus.INTERRUPTED
            record.phase = "compound_command_unknown"
            record.error = {
                "code": "COMMAND_UNKNOWN",
                "message": f"双工具底层命令无法确认，禁止重放: {error}",
            }
            record.updated_at = _now()
            self._save(record)
            return record

        cursors = dict(record.result.get("_feedback_cursors") or {})
        appended: list[Mapping[str, Any]] = []
        next_sequence = record.feedback_cursor
        for command_id, values in feedback.items():
            cursor = int(cursors.get(command_id, 0))
            command = next(item for item in commands if item.command_id == command_id)
            for item in values:
                if item.sequence <= cursor:
                    continue
                next_sequence += 1
                appended.append(
                    {
                        "sequence": next_sequence,
                        "status": command.status.value,
                        "phase": command.status.value,
                        "progress": item.progress,
                        "message": item.message or f"{record.task_name}: {command_id}",
                        "severity": "info",
                        "measurements": {"command_id": command_id},
                        "observations": [],
                        "evidence_refs": [],
                        "timestamp": item.observed_at.isoformat(),
                    }
                )
                cursor = item.sequence
            cursors[command_id] = cursor
        if appended:
            record.feedback = (*record.feedback, *appended)
            record.feedback_cursor = next_sequence
        record.result = {**record.result, "_feedback_cursors": cursors}
        record.progress = sum(item.progress for item in commands) / len(commands)
        record.status = _combined_execution_status(
            tuple(item.status for item in commands)
        )
        self._publish_success_result(record)
        record.phase = record.status.value
        if record.status in {ExecutionStatus.FAILED, ExecutionStatus.INTERRUPTED}:
            record.error = {
                "code": "ROBOT_COMMAND_FAILED"
                if record.status is ExecutionStatus.FAILED
                else "ROBOT_COMMAND_INTERRUPTED",
                "message": next(
                    (item.reason for item in commands if item.reason),
                    "双工具底层命令未能完成",
                ),
            }
        record.updated_at = _now()
        self._save(record)
        return record

    def stop(self, sdk: R1ProSDK, invocation_id: str, reason: str) -> ExecutionRecord:
        with self._lock:
            record = self._require(invocation_id)
            if record.status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.STOPPED,
            }:
                return record
            if not record.command_id:
                record.status = ExecutionStatus.STOPPED
                record.phase = "stopped_before_device_command"
                self._save(record)
                return record
            record.status = ExecutionStatus.STOPPING
            record.phase = "stopping"
            record.updated_at = _now()
            self._save(record)

            command_ids = record.command_ids or (record.command_id,)
            if len(command_ids) > 1:
                return self._stop_many_locked(sdk, record, command_ids, reason)

            try:
                command = sdk.safety.stop_and_hold(record.command_id)
                holding = sdk.state.snapshot().in_hold
            except (KeyError, RobotSDKError) as error:
                command = None
                holding = False
                failure = str(error)
            else:
                failure = command.reason or "未取得底层停止和 hold 证据"
            confirmed = (
                command is not None
                and command.status
                not in {
                    CommandState.ACCEPTED,
                    CommandState.RUNNING,
                    CommandState.INTERRUPTED,
                    CommandState.UNKNOWN,
                }
                and holding
            )
            record.status = (
                ExecutionStatus.STOPPED if confirmed else ExecutionStatus.INTERRUPTED
            )
            record.phase = "held" if confirmed else "stop_unconfirmed"
            record.progress = (
                command.progress if command is not None else record.progress
            )
            record.feedback = (
                *record.feedback,
                {
                    "sequence": record.feedback_cursor + 1,
                    "status": record.status.value,
                    "phase": record.phase,
                    "progress": record.progress,
                    "message": "Robot SDK 已返回停止和 hold 结果",
                    "severity": "info" if confirmed else "critical",
                    "measurements": {"stop_reason": reason},
                    "observations": [],
                    "evidence_refs": [],
                    "timestamp": _now().isoformat(),
                },
            )
            record.feedback_cursor += 1
            record.result = (
                {
                    "stop_evidence": {
                        "command_status": command.status.value
                        if command is not None
                        else "unknown",
                        "holding": holding,
                        "reason": reason,
                    }
                }
                if confirmed
                else {}
            )
            record.error = (
                None
                if confirmed
                else {
                    "code": "STOP_UNCONFIRMED",
                    "message": failure,
                }
            )
            record.updated_at = _now()
            self._save(record)
            return record

    def _stop_many_locked(
        self,
        sdk: R1ProSDK,
        record: ExecutionRecord,
        command_ids: tuple[str, ...],
        reason: str,
    ) -> ExecutionRecord:
        try:
            commands = tuple(
                sdk.safety.stop_and_hold(command_id) for command_id in command_ids
            )
            holding = sdk.state.snapshot().in_hold
        except (KeyError, RobotSDKError) as error:
            commands = ()
            holding = False
            failure = str(error)
        else:
            failure = next(
                (item.reason for item in commands if item.reason),
                "未取得双工具停止和 hold 证据",
            )
        inactive = bool(commands) and all(
            item.status not in {CommandState.ACCEPTED, CommandState.RUNNING}
            for item in commands
        )
        confirmed = inactive and holding
        record.status = (
            ExecutionStatus.STOPPED if confirmed else ExecutionStatus.INTERRUPTED
        )
        record.phase = "held" if confirmed else "stop_unconfirmed"
        record.result = (
            {
                "stop_evidence": {
                    "command_statuses": {
                        item.command_id: item.status.value for item in commands
                    },
                    "holding": holding,
                    "reason": reason,
                }
            }
            if confirmed
            else {}
        )
        record.error = (
            None if confirmed else {"code": "STOP_UNCONFIRMED", "message": failure}
        )
        record.updated_at = _now()
        self._save(record)
        return record

    def _load(self) -> None:
        for (payload,) in self._db.execute("SELECT payload FROM ability_executions"):
            value = json.loads(payload)
            value["status"] = ExecutionStatus(value["status"])
            value["feedback"] = tuple(value.get("feedback") or ())
            value["observations"] = tuple(value.get("observations") or ())
            value["command_ids"] = tuple(value.get("command_ids") or ())
            value["request"] = dict(value.get("request") or {})
            value["updated_at"] = datetime.fromisoformat(value["updated_at"])
            self._records[value["invocation_id"]] = ExecutionRecord(**value)

    def _save(self, record: ExecutionRecord) -> None:
        payload = {
            **record.__dict__,
            "status": record.status.value,
            "request": dict(record.request),
            "feedback": list(record.feedback),
            "observations": list(record.observations),
            "updated_at": record.updated_at.isoformat(),
        }
        self._db.execute(
            "INSERT INTO ability_executions(invocation_id, payload) VALUES(?, ?) "
            "ON CONFLICT(invocation_id) DO UPDATE SET payload=excluded.payload",
            (
                record.invocation_id,
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )
        self._db.commit()

    @staticmethod
    def _publish_success_result(record: ExecutionRecord) -> None:
        """只在所有底层命令确实成功后公开结果模板。

        私有字段仍随执行记录持久化，保证 Server 或 Ability 重启后可以继续
        对账；失败、停止和 interrupted 不会携带任何尚未兑现的成功声明。
        """

        if record.status is not ExecutionStatus.SUCCEEDED:
            return
        success_result = record.result.get("_success_result")
        if isinstance(success_result, Mapping):
            record.result = {**record.result, **dict(success_result)}

    def _require(self, invocation_id: str) -> ExecutionRecord:
        record = self._records.get(invocation_id)
        if record is None:
            raise KeyError(f"未知 invocation_id: {invocation_id}")
        return record


def _execution_status(status: CommandState) -> ExecutionStatus:
    if status is CommandState.ACCEPTED:
        return ExecutionStatus.ACCEPTED
    if status is CommandState.RUNNING:
        return ExecutionStatus.RUNNING
    if status is CommandState.SUCCEEDED:
        return ExecutionStatus.SUCCEEDED
    if status is CommandState.FAILED:
        return ExecutionStatus.FAILED
    if status in {CommandState.STOPPED, CommandState.CANCELLED}:
        return ExecutionStatus.STOPPED
    return ExecutionStatus.INTERRUPTED


def _combined_execution_status(states: tuple[CommandState, ...]) -> ExecutionStatus:
    """只有所有底层工具命令成功时，复合 Ability 才能报告 succeeded。"""

    if any(state is CommandState.UNKNOWN for state in states):
        return ExecutionStatus.INTERRUPTED
    if any(state is CommandState.FAILED for state in states):
        return ExecutionStatus.FAILED
    if all(state is CommandState.SUCCEEDED for state in states):
        return ExecutionStatus.SUCCEEDED
    if all(state in {CommandState.STOPPED, CommandState.CANCELLED} for state in states):
        return ExecutionStatus.STOPPED
    if any(state is CommandState.RUNNING for state in states):
        return ExecutionStatus.RUNNING
    if any(state is CommandState.ACCEPTED for state in states):
        return ExecutionStatus.ACCEPTED
    return ExecutionStatus.INTERRUPTED
