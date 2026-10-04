"""Ability 处理器只共享执行登记，不共享业务判断。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from semantic_robot_sdk_core import Command
from semantic_robot_sdk_r1pro import R1ProSDK

from ..execution import ExecutionRecord, ExecutionRegistry, ExecutionStatus
from ..models import ModelProfileRegistry


class AbilityHandler:
    def __init__(
        self,
        sdk: R1ProSDK,
        executions: ExecutionRegistry,
        models: ModelProfileRegistry | None = None,
        artifact_exchange_root: str | Path | None = None,
    ) -> None:
        self.sdk = sdk
        self.executions = executions
        self.models = models
        self.artifact_exchange_root = (
            Path(artifact_exchange_root).resolve()
            if artifact_exchange_root is not None
            else None
        )

    def execute(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
    ) -> ExecutionRecord:
        raise NotImplementedError

    def after_refresh(self, record: ExecutionRecord) -> ExecutionRecord:
        return record

    def register_command(
        self,
        command: Command,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
        *,
        resource: str,
        result: Mapping[str, Any] | None = None,
    ) -> ExecutionRecord:
        record = self.executions.register(
            command,
            digest,
            invocation_id,
            task_name,
            operation=task_name,
            resource=resource,
            request=data,
            result_template=result,
        )
        return self.after_refresh(
            self.executions.refresh(self.sdk, record.invocation_id)
        )

    def register_commands(
        self,
        commands: tuple[Command, ...],
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
        *,
        resource: str,
        result: Mapping[str, Any] | None = None,
    ) -> ExecutionRecord:
        if len(commands) == 1:
            return self.register_command(
                commands[0],
                task_name,
                data,
                invocation_id,
                digest,
                resource=resource,
                result=result,
            )
        record = self.executions.register_many(
            commands,
            digest,
            invocation_id,
            task_name,
            operation=task_name,
            resource=resource,
            request=data,
            result_template=result,
        )
        return self.after_refresh(
            self.executions.refresh(self.sdk, record.invocation_id)
        )

    def succeeded(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
        result: Mapping[str, Any],
        observations: tuple[Mapping[str, Any], ...] = (),
    ) -> ExecutionRecord:
        return self.executions.register_succeeded(
            task_name,
            digest,
            result,
            invocation_id,
            observations,
            data,
        )

    def failed(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
        code: str,
        message: str,
    ) -> ExecutionRecord:
        return self.executions.register_terminal(
            task_name,
            digest,
            ExecutionStatus.FAILED,
            {},
            invocation_id,
            error={"code": code, "message": message},
            request=data,
        )
