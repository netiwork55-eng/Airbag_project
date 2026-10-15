// Run actual UI functions against minimal DOM fixtures; not a browser/accuracy test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(require('node:path').join(__dirname,'../static/inspection.js'),'utf8');
const line=prefix=>source.split('\n').find(s=>s.startsWith(prefix));
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,{value:'',checked:false,disabled:false,hidden:false});return elements.get(id);};
const ctx={$,structuredClone,reference:{id:'image-fixture'},areas:[{id:'surface'}],locatorRegion:{},saved:null,
  val:id=>Number($(id).value),selectionVersion:0,renderAreas:()=>{},resetValidation:()=>{},
  referenceImage:async()=>{},library:[],trainIds:[],testIds:[],importedIds:[],markDirty:()=>{},page:()=>{},
  polling:false,activePage:'production',stateClass:()=> 'neutral',recipes:[],engineVersion:'fixture',
  api:async()=>({reports:[],runtime:{state:'UNABLE_TO_INSPECT',running:false,reasons:[]},production:null,
    busy:true,external_runtime:{pid:123,purpose:'Terminal camera'},capabilities:{basler_sdk:false}})};
vm.createContext(ctx);
vm.runInContext(line('function config()'),ctx);
assert.equal(ctx.config().validation.min_defect_recall,null);
$('min-defect-recall').value='97.5';
assert.equal(ctx.config().validation.min_defect_recall,.975);
const old=ctx.config();delete old.validation.min_defect_recall;
ctx.recipes=[{id:'old',hash:'old-hash',config:old}];ctx.library=[ctx.reference];
(async()=>{
  vm.runInContext(line('async function loadRecipe('),ctx);
  await ctx.loadRecipe('old');
  assert.equal($('min-defect-recall').value,''); // Old recipe is not silently assigned a target.
  ctx.recipes[0].config.validation.min_defect_recall=.975;
  await ctx.loadRecipe('old');
  assert.equal($('min-defect-recall').value,97.5);
  ctx.bind=(id,fn)=>fn();
  vm.runInContext(line("bind('new-product'"),ctx);
  assert.equal($('min-defect-recall').value,'');
  vm.runInContext(line('async function poll()'),ctx);
  await ctx.poll();
  assert.equal($('live-state').textContent,'External runtime');
  assert.match($('live-reason').textContent,/Terminal camera/);
  for(const id of ['start','save','capture-image','capture-test','validate'])assert.equal($(id).disabled,true);
  console.log('PASS: explicit defect recall round trip, old/new product unset, external owner blocks conflicting UI actions');
})().catch(e=>{console.error(e);process.exitCode=1});
