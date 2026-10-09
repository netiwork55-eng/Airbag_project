// Regression for the retained Core polling behaviour; the retired training page is gone.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),test=require('node:test');
const source=fs.readFileSync(require('node:path').join(__dirname,'../static/inspection.js'),'utf8');
const poll=source.split('\n').find(s=>s.startsWith('async function poll()'));
function env(){const nodes=new Map();const $=id=>{if(!nodes.has(id))nodes.set(id,{});return nodes.get(id);};const ctx={$ ,polling:false,recipes:[],stateClass:()=>'',activePage:'setup'};vm.createContext(ctx);vm.runInContext(poll,ctx);return ctx;}
test('Core polling is single flight and failure cannot leave a live PASS',async()=>{
 const ctx=env();let release,calls=0;ctx.api=async()=>{calls++;await new Promise(r=>release=r);throw Error('offline');};
 const first=ctx.poll();await ctx.poll();assert.equal(calls,1);release();await first;
 assert.equal(ctx.polling,false);assert.equal(ctx.$('live-state').textContent,'UNABLE_TO_INSPECT');assert.equal(ctx.$('live-image').hidden,true);assert.equal(ctx.$('start').disabled,true);
});
test('External terminal ownership blocks conflicting operations without showing fake Production',async()=>{
 const ctx=env();ctx.api=async()=>({runtime:{state:'UNABLE_TO_INSPECT',running:false},production:null,busy:true,external_runtime:{purpose:'Terminal Candidate test',pid:42},capabilities:{basler_sdk:false}});
 await ctx.poll();assert.equal(ctx.$('live-state').textContent,'External runtime');
 for(const id of ['save','capture-image','capture-test','validate','train','start'])assert.equal(ctx.$(id).disabled,true);
});
