// Regression for the retained Core polling behaviour; the retired training page is gone.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),test=require('node:test');
const source=fs.readFileSync(require('node:path').join(__dirname,'../static/inspection.js'),'utf8');
const poll=source.split('\n').find(s=>s.startsWith('async function poll()'));
function env(){const nodes=new Map();const $=id=>{if(!nodes.has(id))nodes.set(id,{});return nodes.get(id);};const ctx={$ ,polling:false,recipes:[],report:null,dirty:false,engineVersion:'current',stateClass:s=>s==='PASS'?'pass':'unable',activePage:'setup'};vm.createContext(ctx);vm.runInContext(poll,ctx);return ctx;}
test('Core polling is single flight and failure cannot leave a live PASS',async()=>{
 const ctx=env();let release,calls=0;ctx.api=async()=>{calls++;await new Promise(r=>release=r);throw Error('offline');};
 const first=ctx.poll();await ctx.poll();assert.equal(calls,1);release();await first;
 assert.equal(ctx.polling,false);assert.equal(ctx.$('live-state').textContent,'UNABLE_TO_INSPECT');assert.equal(ctx.$('live-image').hidden,true);assert.equal(ctx.$('start').disabled,true);
});
function status(overrides={}){return {runtime:{state:'PASS',running:true,reasons:[]},production:{recipe_id:'recipe',recipe_hash:'hash',engine_version:'current',previous:{}},engine_version:'current',busy:false,capabilities:{basler_sdk:false},...overrides};}
test('Lost status clears stale green badges and blocks changes until a fresh status arrives',async()=>{
 const ctx=env();ctx.api=async()=>status();await ctx.poll();
 assert.equal(ctx.$('live-state').className,'badge pass');
 assert.equal(ctx.$('active-label').className,'badge pass');
 ctx.$('activate').disabled=false;ctx.api=async()=>{throw Error('offline');};await ctx.poll();
 assert.equal(ctx.$('live-state').className,'badge unable');
 assert.equal(ctx.$('active-label').className,'badge neutral');
 assert.equal(ctx.$('live-empty').hidden,false);
 for(const id of ['start','rollback','save','train','train-labelled','capture-image','capture-test','validate','activate'])assert.equal(ctx.$(id).disabled,true,id);
 // Keep Stop available if it was previously available; offline does not prove the worker stopped.
 assert.equal(ctx.$('stop').disabled,false);
 ctx.api=async()=>status({runtime:{state:'UNABLE_TO_INSPECT',running:false}});await ctx.poll();
 assert.equal(ctx.$('save').disabled,false);assert.equal(ctx.$('start').disabled,false);
 assert.equal(ctx.$('activate').disabled,true,'No validated report must not become activatable on recovery');
});
test('A production pointer from an older engine is not shown as ready',async()=>{
 const ctx=env();ctx.api=async()=>status({production:{recipe_hash:'oldhash',engine_version:'old'}});await ctx.poll();
 assert.equal(ctx.$('active-label').className,'badge review');assert.equal(ctx.$('start').disabled,true);
});
test('Polling preserves activation safety across external ownership and recovery',async()=>{
 const ctx=env();ctx.report={eligible:true,engine_version:'current'};
 ctx.api=async()=>status({busy:true});await ctx.poll();assert.equal(ctx.$('activate').disabled,true);
 ctx.api=async()=>status();await ctx.poll();assert.equal(ctx.$('activate').disabled,false);
 ctx.dirty=true;await ctx.poll();assert.equal(ctx.$('activate').disabled,true);
 ctx.dirty=false;ctx.report.engine_version='old';await ctx.poll();assert.equal(ctx.$('activate').disabled,true);
});
test('External terminal ownership blocks conflicting operations without showing fake Production',async()=>{
 const ctx=env();ctx.api=async()=>({runtime:{state:'UNABLE_TO_INSPECT',running:false},production:null,busy:true,external_runtime:{purpose:'Terminal camera',pid:42},capabilities:{basler_sdk:false}});
 await ctx.poll();assert.equal(ctx.$('live-state').textContent,'External runtime');
 for(const id of ['save','capture-image','capture-test','validate','train','start'])assert.equal(ctx.$(id).disabled,true);
});
