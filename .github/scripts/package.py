# SPDX-License-Identifier: Apache-2.0
import hashlib, json, os, shutil, subprocess, tarfile, zipfile
from pathlib import Path
root=Path.cwd(); out=root/'.output/release'; out.mkdir(parents=True,exist_ok=True)
payload=root/'.output/payload'; payload.mkdir(parents=True,exist_ok=True)
component=os.environ['COMPONENT']; tag=os.environ.get('TARGET_TAG','')
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if tag and subprocess.check_output(['git','rev-parse',f'refs/tags/{tag}^{{commit}}'],text=True).strip()!=sha: raise SystemExit('Tag/source mismatch')
platform='linux-x86_64' if component in ('semantic-deployment','ability-runtime','mujoco-runtime') else 'any'
meta=dict(component=component,tag=tag,source_commit=sha,build_recipe_commit=os.environ.get('GITHUB_SHA',''),platform=platform,runner='ubuntu-24.04',workflow_run=f"https://github.com/{os.environ.get('GITHUB_REPOSITORY','')}/actions/runs/{os.environ.get('GITHUB_RUN_ID','')}",validation_scope='Component tests and archive checks; excludes GPU, hardware and full-stack acceptance.')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
for p in root.iterdir():
 if p.is_file() and (p.name.startswith(('LICENSE','NOTICE')) or p.name in ('ASSET_PROVENANCE.md','EXTERNAL_MODELS.md')):shutil.copy2(p,payload/p.name)
for directory in ('third-party-licenses',):
 if (root/directory).is_dir():shutil.copytree(root/directory,payload/directory)
for p in (root/'dist').glob('*'):
 if p.suffix=='.whl':
  with zipfile.ZipFile(p) as z:
   if z.testzip():raise SystemExit('Corrupt wheel')
 if p.is_file() and (p.suffix=='.whl' or p.name.endswith('.tar.gz')):shutil.copy2(p,out/p.name)
for p in payload.rglob('*'):
 if p.is_symlink():raise SystemExit(f'Unexpected symlink: {p}')
 if p.is_file():
  with p.open('rb') as f:
   if f.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):raise SystemExit(f'Unresolved LFS pointer: {p}')
if (root/'.output/dependencies.json').exists():meta['dependencies']=json.loads((root/'.output/dependencies.json').read_text())
(payload/'release.json').write_text(json.dumps(meta,indent=2)+'\n')
with tarfile.open(out/f"{component}-{tag or 'ci-'+sha[:12]}-{platform}.tar.gz",'w:gz',compresslevel=3) as tar:
 for p in sorted(payload.iterdir()):tar.add(p,arcname=p.name)
(out/'release.json').write_text(json.dumps(meta,indent=2)+'\n')
(out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.name!='SHA256SUMS'))
