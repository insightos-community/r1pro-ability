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

"""模型默认值和允许列表。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ModelProfileError(ValueError):
    """模型未启用、类型不匹配或配置不可读。"""


@dataclass(frozen=True)
class ModelProfile:
    name: str
    ability_role: str
    provider: str
    settings: Mapping[str, Any]


class ModelProfileRegistry:
    def __init__(
        self, defaults: Mapping[str, str], profiles: Mapping[str, ModelProfile]
    ) -> None:
        self._defaults = dict(defaults)
        self._profiles = dict(profiles)

    @classmethod
    def load(cls, path: str | Path) -> "ModelProfileRegistry":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelProfileError(f"无法读取模型配置: {path}") from exc
        profiles = {
            name: ModelProfile(
                name=name,
                ability_role=str(value["ability_role"]),
                provider=str(value["provider"]),
                settings=dict(value.get("settings") or {}),
            )
            for name, value in (data.get("profiles") or {}).items()
            if bool(value.get("enabled", True))
        }
        return cls(data.get("defaults") or {}, profiles)

    def resolve(self, ability_role: str, requested: str | None) -> ModelProfile:
        profile_name = requested or self._defaults.get(ability_role)
        if not profile_name:
            raise ModelProfileError(f"{ability_role} 没有默认模型")
        profile = self._profiles.get(profile_name)
        if profile is None:
            raise ModelProfileError(f"模型不在允许列表: {profile_name}")
        if profile.ability_role != ability_role:
            raise ModelProfileError(f"模型 {profile_name} 不适用于 {ability_role}")
        return profile
