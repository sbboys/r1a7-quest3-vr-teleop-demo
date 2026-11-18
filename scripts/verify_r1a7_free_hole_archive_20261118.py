#!/usr/bin/env python3
"""Verify research archive without importing models, cameras or control programs."""
from pathlib import Path
import difflib,hashlib,json,re,subprocess,tarfile,xml.etree.ElementTree as ET,zipfile
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'r1a7_threading_project/docs/source_records/2026-11-18_free_hole'
E=ROOT/'r1a7_threading_project/docs/experiments/2026-11-18_free_hole'
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 m=json.loads((S/'manifest.json').read_text());artifacts=json.loads((E/'artifact_manifest.json').read_text())
 for x in artifacts+m['files']:
  b=(ROOT/x['path']).read_bytes();assert digest(b)==x['sha256'],x['path'];assert len(b)==x['bytes']
 expected=json.loads((S/'expected_historical_compile_failures.json').read_text());fail=[];good=0
 for x in m['files']:
  p=ROOT/x['path']
  if '.py' not in p.name or p.suffix=='.gz':continue
  try:compile(p.read_bytes(),str(p),'exec');good+=1
  except (SyntaxError,ValueError) as e:fail.append({'path':x['path'],'type':type(e).__name__,'message':str(e)})
 assert fail==expected,'Unexpected source/backup compile failure'
 for x in m['files']:
  if x['role'] in ('current_research_source','normal_research_visual'):assert x['path'] not in {f['path'] for f in fail}
 doc=next(S.glob('*.docx'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
 with zipfile.ZipFile(doc) as z:
  r=ET.fromstring(z.read('word/document.xml'))
  paras=[''.join((t.text or '') if t.tag.endswith('}t') else ('\n' if t.tag.endswith('}br') else '\t') for t in p.iter() if t.tag.rsplit('}',1)[-1] in ('t','br','tab')).rstrip() for p in r.findall('.//w:p',ns)]
  assert len(paras)==m['source_docx_paragraphs'];transcript=(S/'worklog_fulltext_zh.md').read_text();assert all(p in transcript for p in paras if p)
  media=[n for n in z.namelist() if n.startswith('word/media/')]
  for n in media:assert z.read(n)==(S/'assets'/Path(n).name).read_bytes()
 backup=S/'safety_backups/pre_lower_branch_free_20261118_005624'; checksum=(backup/'source.sha256').read_text().splitlines();count=0
 with tarfile.open(backup/'r1a7_pre_lower_branch_free_source_snapshot.tar.gz') as t:
  for row in checksum:
   wanted,path=row.split(None,1);path=path.strip();b=t.extractfile(path).read();assert digest(b)==wanted
   assert b==(backup/'files'/Path(path).name).read_bytes();count+=1
 base=S/'safety_backups/pre_free_hole_live_20261118_095120/live_grasp_to_insertion_visual_v3.py';assert digest(base.read_bytes())==m['document_visual_sha256']
 vis=next(ROOT/x['path'] for x in m['files'] if x['role']=='current_research_source' and x['path'].endswith('live_grasp_to_insertion_visual_v3.py'));txt=vis.read_text();assert digest(vis.read_bytes())==m['current_visual_sha256']
 assert 'FREE_CONTINUOUS_HOLE_LIVE_RELOCALIZATION_V3' in txt and 'INSERTION_FREE_LAST_TIP_HOLD_S = 0.75' in txt
 assert 'phase\n                    in (\n                        PHASE_WAIT_HOLE,\n                        PHASE_INSERTION,' in txt
 assert 'camera\n                            !=\n                            "left_wrist"' in txt
 for rel in ['vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py','vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py','transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py']:
  path='r1a7_threading_project/'+rel;assert subprocess.check_output(['git','-C',str(ROOT),'show',m['base_commit']+':'+path])==(ROOT/path).read_bytes()
 oldpath='r1a7_threading_project/docs/source_records/2026-11-17_insertion_debug/research_code_snapshot/current/'+str(vis.relative_to(S/'research_code_snapshot/current'))
 old=subprocess.check_output(['git','-C',str(ROOT),'show',m['base_commit']+':'+oldpath]).decode()
 first=json.loads((S/'diffs/live_grasp_to_insertion_visual_v3.py.diff.json').read_text())
 assert first['diff']==''.join(difflib.unified_diff(old.splitlines(True),txt.splitlines(True),fromfile=m['base_commit']+'/'+oldpath,tofile=str(vis.relative_to(S/'research_code_snapshot/current'))))
 second=json.loads((S/'diffs/visual_document_baseline_to_current.diff.json').read_text())
 assert second['diff']==''.join(difflib.unified_diff(base.read_text().splitlines(True),txt.splitlines(True),fromfile='document_baseline_b9a12787',tofile='current_visual'))
 links=0
 for p in [S/'README.md',ROOT/'r1a7_threading_project/docs/r1a7_free_hole_archive_2026-11-18_zh.md',ROOT/'r1a7_threading_project/docs/r1a7_free_hole_sop_2026-11-18_zh.md']:
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if '://' not in link:assert (p.parent/link.split('#')[0]).exists(),link;links+=1
  for block in re.findall(r'```bash\n(.*?)```',p.read_text(),re.S):subprocess.run(['bash','-n'],input=block,text=True,check=True)
 out={'source_files':len(m['files']),'artifact_hashes':len(artifacts),'successful_source_compile':good,'expected_historical_compile_failures':len(fail),'docx_paragraphs':len(paras),'original_images':len(media),'tar_source_checksums':count,'local_links':links,'current_V3_source_present':True,'console_output_chain_unchanged':True,'cited_trace_available':False,'runtime_executed':False,'hardware_executed':False}
 (E/'archive_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
