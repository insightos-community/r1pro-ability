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
