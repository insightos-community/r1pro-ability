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

ROBOT_SDK_PATH ?= ../semantic-robot-sdk
export PYTHONPATH := $(abspath $(ROBOT_SDK_PATH)/packages/core/src):$(abspath $(ROBOT_SDK_PATH)/packages/r1pro/src):$(CURDIR):$(PYTHONPATH)
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: check test build

check:
	$(PYTHON) -m compileall -q r1pro_abilities abilities tests

test: check
	$(PYTHON) -m unittest discover -s tests -v

build: test
	uv build --wheel
