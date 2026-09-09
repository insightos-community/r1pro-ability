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

"""SensorCapture Ability：输出可由 Pilot 发布的真实 RGBD Artifact 候选。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..ability_utils import observation
from ..execution import ExecutionRecord
from .base import AbilityHandler


class SensorCaptureHandler(AbilityHandler):
    def execute(
        self,
        task_name: str,
        data: Mapping[str, Any],
        invocation_id: str,
        digest: str,
    ) -> ExecutionRecord:
        if task_name != "CaptureRGBD":
            raise ValueError(f"SensorCapture 不支持 Task: {task_name}")
        if self.artifact_exchange_root is None:
            return self.failed(
                task_name,
                data,
                invocation_id,
                digest,
                "ARTIFACT_EXCHANGE_UNAVAILABLE",
                "实例没有配置 Ability/Pilot Artifact 交换目录",
            )

        candidates = []
        observations = []
        for sensor_id in data["sensor_ids"]:
            frame = self.sdk.sensors.latest(str(sensor_id))
            candidate_id = (
                f"frame:{self.sdk.robot_id}:{sensor_id}:"
                f"{frame.generation}:{frame.sequence}"
            )
            media_type = _media_type(frame.encoding)
            exchange_path = _publish_frame(
                self.artifact_exchange_root,
                invocation_id,
                str(sensor_id),
                frame.generation,
                frame.sequence,
                frame.encoding,
                frame.payload,
            )
            candidate = {
                "candidate_id": candidate_id,
                "sensor_id": sensor_id,
                "generation": frame.generation,
                "sequence": frame.sequence,
                "frame_id": frame.frame_id,
                "encoding": frame.encoding,
                "media_type": media_type,
                "width": frame.width,
                "height": frame.height,
                "observed_at": frame.observed_at.isoformat(),
                "payload_size": len(frame.payload)
                if isinstance(frame.payload, bytes)
                else None,
                "exchange_path": exchange_path,
                "publication_state": "ready_for_pilot",
            }
            candidates.append(candidate)
            observations.append(
                observation(
                    "sensor.frame",
                    f"robot-sdk://{self.sdk.robot_id}/sensor/{sensor_id}",
                    self.sdk.robot_id,
                    candidate,
                    revision=f"{frame.generation}:{frame.sequence}",
                    frame_id=frame.frame_id,
                )
            )

        # 这里只返回交换目录内的相对句柄，绝不伪造 artifact:// 引用，也不把
        # 绝对文件路径暴露给 Worker。Pilot 校验路径后读取并发布，最终才产生
        # Server ArtifactRef。
        return self.succeeded(
            task_name,
            data,
            invocation_id,
            digest,
            {"artifact_candidates": candidates, "artifact_refs": []},
            tuple(observations),
        )


def _media_type(encoding: str) -> str:
    if encoding == "jpeg":
        return "image/jpeg"
    if encoding.startswith("png"):
        return "image/png"
    if encoding.startswith("float32"):
        return "application/octet-stream"
    if encoding == "json":
        return "application/json"
    return "application/octet-stream"


def _publish_frame(
    root: Path,
    invocation_id: str,
    sensor_id: str,
    generation: str | int,
    sequence: int,
    encoding: str,
    payload: bytes | Mapping[str, Any],
) -> str:
    """原子写入真实传感器帧并返回相对根目录的句柄。

    invocation_id 与 sensor_id 都可能来自外部输入，因此文件名使用摘要而
    不是直接拼接，避免路径穿越。临时文件与目标位于同一目录，os.replace
    保证 Pilot 永远不会读到半个 RGB/Depth 帧。
    """

    invocation_key = hashlib.sha256(invocation_id.encode("utf-8")).hexdigest()[:20]
    sensor_key = hashlib.sha256(sensor_id.encode("utf-8")).hexdigest()[:16]
    generation_key = hashlib.sha256(str(generation).encode("utf-8")).hexdigest()[:12]
    directory = root / "captures" / invocation_key
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{sensor_key}-{generation_key}-{sequence}.{_extension(encoding)}"
    target = directory / filename
    temporary = target.with_suffix(target.suffix + ".partial")
    content = (
        payload
        if isinstance(payload, bytes)
        else json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    temporary.write_bytes(content)
    os.replace(temporary, target)
    return target.relative_to(root).as_posix()


def _extension(encoding: str) -> str:
    if encoding == "jpeg":
        return "jpg"
    if encoding.startswith("png"):
        return "png"
    if encoding == "json":
        return "json"
    return "bin"
