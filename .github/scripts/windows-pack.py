"""Build seven Windows Ability archives using the installed native scaffold."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import yaml

root = Path.cwd()
output = root/'.output/windows'
output.mkdir(parents=True, exist_ok=True)
launcher = Path(os.environ['SEMANTIC_WINDOWS_ABILITY_LAUNCHER'])
assert launcher.read_bytes()[:2] == b'MZ'
projects = sorted((root/'abilities').glob('*/ability.manifest.yaml'))
assert len(projects) == 7
records = []
for manifest in projects:
    project = manifest.parent
    archive = output/(project.name+'.zip')
    subprocess.run([sys.executable, '-m', 'ability_scaffold.cli', 'pack', str(project), '-o', str(archive)], check=True)
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert package.read('bin/ability.exe') == launcher.read_bytes()
        assert all('\\' not in name for name in package.namelist())
        metadata = yaml.safe_load(package.read('package.yaml'))
        assert metadata['arch'] == 'x86_64'
        assert package.read('main.py')
    records.append({'file':archive.name, 'name':metadata['name'], 'version':str(metadata['version']),
                    'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
report = {'platform':'windows-amd64', 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'native_sources':{name:subprocess.check_output(['git','-C','sources/'+name,'rev-parse','HEAD'],text=True).strip()
                            for name in ('scaffold','ability-sdk')},
          'launcher_sha256':hashlib.sha256(launcher.read_bytes()).hexdigest(), 'abilities':records,
          'validation_scope':'Native unit tests and seven package archives; full AbilityFramework/Robot/MuJoCo lifecycle remains separate.'}
(output/'windows-packages.json').write_text(json.dumps(report,indent=2)+'\n')
(output/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n'
    for path in sorted(output.iterdir()) if path.is_file() and path.name != 'SHA256SUMS'))
print('PASS seven Windows Ability packages with the exact native launcher')
