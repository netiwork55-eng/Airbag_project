const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),test=require('node:test');
const source=fs.readFileSync(require('node:path').join(__dirname,'../static/inspection.js'),'utf8');
function environment(){
 const nodes=new Map(),handlers={},calls=[];
 const node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',files:[],disabled:false,innerHTML:'',textContent:'',replaceChildren(){this.innerHTML='';},addEventListener(){},querySelectorAll(){return [];}});return nodes.get(id);};
 const ctx={$:node,esc:s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),bind:(id,fn)=>handlers[id]=fn,message(){},confirm:()=>true,FormData,api:async()=>({training:[]}),fetch:async(path,options)=>{calls.push([path,options]);return {ok:true,json:async()=>({ok:true,folders:[],jobs:[],test_reports:[],models:[],inspection_models:[],runs:[],parts:[],reviews:[]})};}};
 vm.createContext(ctx);vm.runInContext(source.slice(source.indexOf('// Existing data/job/review services')),ctx);
 return {ctx,node,handlers,calls};
}
test('Legacy NG never becomes confirmed; stored human status and HTML are preserved safely',()=>{
 const {ctx}=environment();
 assert.match(ctx.legacyHumanStatus({decision:'NG',status:'pending'}),/ยังไม่ยืนยัน/);
 assert.match(ctx.legacyHumanStatus({decision:'NG',status:'confirmed_ng'}),/ยังไม่ยืนยัน/);
 assert.match(ctx.legacyHumanStatus({decision:'NG',status:'false_alarm',reviewed_at:'2026'}),/false_alarm/);
 const row=ctx.legacyReviewRow({id:'ng_old',decision:'NG',source_name:'<script>alert(1)</script>',status:'pending'});
 assert.ok(row.includes('Machine: NG'));assert.ok(!row.includes('<script>'));assert.ok(row.includes('/api/inspection/reviews/ng_old/source'));
});
test('Dataset metadata saves description only, never label or human confirmation',async()=>{
 const {ctx,node,handlers,calls}=environment();node('dataset-folder').value='real part';node('data-meta-dataset_type').value='Known-good Reference';
 await handlers['data-save-meta']();const [url,options]=calls[0];
 assert.equal(url,'/api/dataset-folders/real%20part/metadata');
 const payload=JSON.parse(options.body);assert.equal(payload.dataset_type,'Known-good Reference');assert.ok(!('label' in payload));assert.ok(!('confirmed_at' in payload));
});
test('Preview ignores stale response after folder changes and Undo uses server token',async()=>{
 const {ctx,node,handlers,calls}=environment();node('dataset-folder').value='A';let resolve;
 ctx.fetch=async path=>({ok:true,json:()=>new Promise(r=>{if(path.endsWith('/preview'))resolve=r;else r({can_undo:true,upload_id:'token',total:1});})});
 const pending=ctx.previewDataset();await new Promise(setImmediate);
 ctx.selectDatasetMetadata();resolve({preview:{images:1,labels:0,samples:[{url:'/old.png'}]}});await pending;
 assert.equal(node('data-preview-list').innerHTML,'');assert.equal(node('data-undo').disabled,true);
 ctx.fetch=async(path,options)=>{calls.push([path,options]);return {ok:true,json:async()=>path.endsWith('/preview')?{preview:{images:1,labels:0,samples:[]}}:path.endsWith('/latest-upload')?{can_undo:true,upload_id:'new-token',total:1}:{folders:[]}};};
 await ctx.previewDataset();assert.equal(node('data-undo').disabled,false);await handlers['data-undo']();
 const undo=calls.find(([p])=>p.endsWith('/undo-upload'));
 assert.deepEqual(JSON.parse(undo[1].body),{folder:'A',upload_id:'new-token'});
});
test('Trash lists existing backend folders and backend errors surface instead of empty success',async()=>{
 const {ctx,node}=environment();ctx.fetch=async()=>({ok:true,json:async()=>({folders:[{id:'trash1',original_name:'part',deleted_at:'2026'}]})});
 await ctx.refreshTrash();assert.ok(node('data-trash-list').innerHTML.includes('data-restore="trash1"'));
 ctx.fetch=async()=>({ok:false,status:409,json:async()=>({error:'busy'})});
 await assert.rejects(()=>ctx.workspaceRequest('/api/dataset-folders'),/busy/);
});
test('Job and model histories stay independently readable; refresh never mutates or trains',async()=>{
 const {ctx,node,calls}=environment();ctx.fetch=async(path,opts)=>{calls.push([path,opts]);return {ok:true,json:async()=>path==='/api/jobs'?{jobs:[{id:'job_old',name:'old'}],test_reports:[{id:'test_abc'}]}:path==='/api/models/registry'?{models:[{name:'old.pt'}],inspection_models:[{id:'model_candidate'}]}:{runs:[]}};};
 await ctx.refreshJobs();assert.ok(node('model-library-list').innerHTML.includes('model_candidate'));assert.ok(node('job-library-list').innerHTML.includes('/api/jobs/test-report/test_abc'));
 assert.ok(calls.every(([p,o])=>!o.method));
});
test('Legacy test uses existing endpoint and renders preview without dumping binary data',async()=>{
 const {ctx,node,handlers,calls}=environment();node('legacy-job-select').value='job_old';node('legacy-job-image').files=[new Blob(['fixture'],{type:'image/png'})];
 ctx.fetch=async(path,opts)=>{calls.push([path,opts]);return {ok:true,json:async()=>({decision:'NG',preview:'data:image/jpeg;base64,AA==',anomaly_map:[[1]],job_id:'job_old',use:'test'})};};
 await handlers['legacy-job-test']();assert.equal(calls[0][0],'/api/jobs/job_old/test');assert.equal(node('legacy-job-preview').hidden,false);assert.ok(!node('legacy-job-result').textContent.includes('base64'));assert.ok(node('legacy-job-result').textContent.includes('ยังไม่ยืนยัน'));
});
