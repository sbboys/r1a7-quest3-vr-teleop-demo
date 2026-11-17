#!/usr/bin/env python3
"""Read archive bytes and compile text only; never import robot/model code."""
from pathlib import Path
import csv,gzip,hashlib,io,json,re,subprocess,xml.etree.ElementTree as ET,zipfile
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'r1a7_threading_project/docs/source_records/2026-11-17_insertion_debug'
E=ROOT/'r1a7_threading_project/docs/experiments/2026-11-17_insertion_debug'
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 m=json.loads((S/'manifest.json').read_text()); artifacts=json.loads((E/'artifact_manifest.json').read_text())
 for x in artifacts+m['files']:
  b=(ROOT/x['path']).read_bytes();assert digest(b)==x['sha256'],x['path'];assert len(b)==x['bytes']
 code=[]
 for p in S.rglob('*'):
  if p.is_file() and (p.name.endswith('.py') or '.py.before_' in p.name or '.py.current_raw_' in p.name):
   compile(p.read_bytes(),str(p),'exec');code.append(str(p.relative_to(ROOT)))
 doc=next(S.glob('*.docx'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
 with zipfile.ZipFile(doc) as z:
  r=ET.fromstring(z.read('word/document.xml'))
  paras=[''.join((t.text or '') if t.tag.endswith('}t') else ('\n' if t.tag.endswith('}br') else '\t') for t in p.iter() if t.tag.rsplit('}',1)[-1] in ('t','br','tab')).rstrip() for p in r.findall('.//w:p',ns)]
  assert len(paras)==m['source_docx_paragraphs']
  transcript=(S/'worklog_fulltext_zh.md').read_text();assert all(p in transcript for p in paras if p)
  for n in z.namelist():
   if n.startswith('word/media/'):assert z.read(n)==(S/'assets'/Path(n).name).read_bytes()
 trace=next(E.glob('*.csv.gz'));raw=gzip.decompress(trace.read_bytes());assert digest(raw)==m['trace']['source_sha256']
 rows=list(csv.DictReader(io.StringIO(raw.decode())));assert len(rows)==3428 and len(rows[0])==18
 checks=0
 for p in [S/'README.md',ROOT/'r1a7_threading_project/docs/r1a7_insertion_debug_archive_2026-11-17_zh.md',ROOT/'r1a7_threading_project/docs/r1a7_insertion_debug_sop_2026-11-17_zh.md']:
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' not in link:assert (p.parent/link.split('#')[0]).exists(),link;checks+=1
 # Stable console output-chain paths remain byte-identical to prearchive baseline.
 for rel in ['vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py','vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py','transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py']:
  path='r1a7_threading_project/'+rel
  old=subprocess.check_output(['git','-C',str(ROOT),'show',m['base_commit']+':'+path]);assert old==(ROOT/path).read_bytes()
 for p in [ROOT/'r1a7_threading_project/docs/r1a7_insertion_debug_sop_2026-11-17_zh.md']:
  for block in re.findall(r'```bash\n(.*?)```',p.read_text(),re.S):subprocess.run(['bash','-n'],input=block,text=True,check=True)
 out={'source_files':len(m['files']),'artifact_hashes':len(artifacts),'compiled_source_and_backups':len(code),'docx_paragraphs':len(paras),'trace_rows':len(rows),'trace_columns':18,'local_links':checks,'console_output_chain_unchanged':True,'matching_full_chain_available':False,'runtime_executed':False,'hardware_executed':False}
 (E/'archive_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
