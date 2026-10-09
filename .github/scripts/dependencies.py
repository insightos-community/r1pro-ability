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

import argparse,hashlib,json,subprocess
from pathlib import Path
pins=[('insightos-community/Ability-SDK-Python', 'v0.4.0-insightos.2026.2', '3a90b6d94ea6babb890ffcd285ba55e5fef6fc8c'), ('insightos-community/robot-sdk', 'v0.4.0-insightos.2026.2', '59a1a8364c3d0330f1802c020e37544e0dbfa7c5'), ('insightos-community/ability-scaffold', 'v1.2.0-insightos.2026.2', 'f85b9cce4f0040671fa74ebf0a587b7e4c65a7df')]
parser=argparse.ArgumentParser()
parser.add_argument('--robot-sdk-only', action='store_true')
if parser.parse_args().robot_sdk_only:
 pins=[pin for pin in pins if pin[0]=='insightos-community/robot-sdk']
records=[]
for repo,tag,sha in pins:
 dest=Path('.output/deps')/repo.split('/')[-1];dest.mkdir(parents=True)
 subprocess.run(['gh','release','download',tag,'--repo',repo,'--dir',str(dest),'--pattern','*.whl','--pattern','SHA256SUMS','--pattern','release.json'],check=True)
 sums=dict(line.split('  ',1)[::-1] for line in (dest/'SHA256SUMS').read_text().splitlines())
 for p in dest.iterdir():
  if p.name!='SHA256SUMS' and hashlib.sha256(p.read_bytes()).hexdigest()!=sums[p.name]:raise SystemExit('Dependency checksum mismatch')
 meta=json.loads((dest/'release.json').read_text())
 if meta['tag']!=tag or meta['source_commit']!=sha:raise SystemExit('Dependency version mismatch')
 records.append(dict(repository=repo,tag=tag,source_commit=sha,checksums=sums))
Path('.output/dependencies.json').write_text(json.dumps(records,indent=2)+'\n')
