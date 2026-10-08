"""Verify generated WAVs, blind bindings, and ratings-page behavior without listening."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.vc_tournament.audio import digest,stats

def main():
 folder=ROOT/'recordings/v011_post_vc'
 rows=json.loads((folder/'metadata/results.json').read_text('utf-8'))
 manifest=json.loads((folder/'metadata/blind_manifest.json').read_text('utf-8'))
 assert len(manifest)==sum(r['status']=='SUCCESS' for r in rows)
 assert len({r['id'] for r in manifest})==len(manifest)
 baseline={r['source']:r for r in manifest if r['model']=='baseline'}
 assert set(baseline)=={'normal','low','bright','long'}
 for row in manifest:
  raw=stats(row['output']);blind=stats(folder/'blind'/(row['id']+'.wav'))
  assert raw['sha256']==row['normalization']['raw']['sha256']
  assert blind['sha256']==row['normalization']['after']['sha256']
  assert blind['clipping_samples']==0
  assert abs(raw['duration']-stats(baseline[row['source']]['output'])['duration'])<=.1
  if row['model']!='baseline':assert raw['sha256']!=stats(baseline[row['source']]['output'])['sha256']
 html=(folder/'blind/index.html').read_text('utf-8')
 assert '__DATA__' not in html
 script=html.split('<script>',1)[1].split('</script>',1)[0]
 js=folder/'metadata/listening-script.js';js.write_text(script,'utf-8')
 node=shutil.which('node');assert node,'Node required for actual JS validation'
 subprocess.run([node,'--check',str(js)],check=True)
 # Execute against a small DOM shim: verify filtering, ratings, export and import.
 harness=folder/'metadata/listening-test.js'
 harness.write_text(r'''
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert');
class Element{constructor(){this.children=[];this.value='';this.files=[];this.textContent='';}append(...x){this.children.push(...x)}replaceChildren(...x){this.children=x}click(){}}
const elements=new Map();for(const id of ['candidates','status','case-filter','export','import'])elements.set(id,new Element());elements.get('case-filter').value='long';
let exported;const memory=new Map();const sandbox={document:{getElementById:id=>elements.get(id),createElement:()=>new Element()},localStorage:{getItem:k=>memory.get(k),setItem:(k,v)=>memory.set(k,v)},window:{showSaveFilePicker:async()=>({createWritable:async()=>({write:async blob=>{exported=await blob.text()},close:async()=>{}})})},Blob,URL,setTimeout};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),sandbox);
async function test(){
const total=vm.runInContext('data.candidates.length',sandbox),long=vm.runInContext("data.candidates.filter(c=>c.case==='long').length",sandbox);
assert.equal(elements.get('candidates').children.length,long);
const article=elements.get('candidates').children[0],id=vm.runInContext("data.candidates.find(c=>c.case==='long').id",sandbox);
const score=article.children[2].children[0];score.value='5';score.onchange();
const comment=article.children.at(-1);comment.value='声を保てるか確認';comment.oninput();
await elements.get('export').onclick();const data=JSON.parse(exported);assert.equal(data.ratings[id].similarity,'5');assert.equal(data.ratings[id].comment,'声を保てるか確認');
elements.get('case-filter').value='all';elements.get('case-filter').onchange();assert.equal(elements.get('candidates').children.length,total);
await elements.get('import').onchange({target:{files:[{text:async()=>exported}]}});assert.equal(elements.get('candidates').children.length,total);
console.log('Filter + rating persistence + JSON export/import verified (DOM shim; no playback)');
}test().catch(e=>{console.error(e);process.exitCode=1});
''','utf-8')
 subprocess.run([node,str(harness),str(js)],check=True)
 result=dict(status='SUCCESS',candidates=len(manifest),models=sorted({r['model'] for r in manifest}),wav_and_bindings_verified=True,js_syntax_verified=True,ratings_dom_shim_verified=True,human_listening=False,physical_audio_trial=False)
 (folder/'metadata/assets_verification.json').write_text(json.dumps(result,indent=2),'utf-8')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
