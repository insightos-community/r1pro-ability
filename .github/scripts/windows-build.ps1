$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
python .github/scripts/dependencies.py --robot-sdk-only
if ($LASTEXITCODE -ne 0) { throw 'Dependency download failed' }
New-Item -ItemType Directory -Force .output/wheels | Out-Null
$wheelhouse = (Resolve-Path .output/wheels).Path
cl /nologo /std:c++20 /EHsc /MT /O2 /utf-8 sources/scaffold/native/windows/ability.cpp /Fe:.output/ability.exe /link shell32.lib
if ($LASTEXITCODE -ne 0) { throw 'Native launcher build failed' }
$env:SEMANTIC_WINDOWS_ABILITY_LAUNCHER = (Resolve-Path .output/ability.exe).Path
foreach ($project in @('sources/scaffold', 'sources/ability-sdk', '.')) {
  uv build --wheel --project $project --out-dir $wheelhouse
  if ($LASTEXITCODE -ne 0) { throw "Wheel build failed: $project" }
}
uv venv --python 3.13.15 .output/test
if ($LASTEXITCODE -ne 0) { throw 'Test venv failed' }
$wheels = @((Get-ChildItem .output/wheels/*.whl).FullName) + @((Get-ChildItem .output/deps/robot-sdk/*.whl).FullName)
uv pip install --python .output/test/Scripts/python.exe @wheels 'PyYAML==6.0.2' 'pydantic==2.13.4' 'flask==3.1.3' 'requests==2.34.2' 'websockets==17.0.1'
if ($LASTEXITCODE -ne 0) { throw 'Wheel installation failed' }
.output/test/Scripts/python.exe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Native tests failed' }
.output/test/Scripts/python.exe .github/scripts/windows-pack.py
if ($LASTEXITCODE -ne 0) { throw 'Windows packaging failed' }
