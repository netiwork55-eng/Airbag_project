"""Validation never silently tunes thresholds or promotes a recipe."""
import uuid
from collections import Counter
import numpy as np
from .store import atomic_json, now, digest, label_confirmed, group_confirmed
from . import ENGINE_VERSION


def validate(store, pipeline, image_ids, dataset_id=None):
    if not 1 <= len(image_ids) <= 5000 or len(set(image_ids)) != len(image_ids):
        raise ValueError("Choose 1–5000 distinct images")
    recipe = pipeline.recipe
    used = {recipe.reference_id}
    groups = {store.read("images",recipe.reference_id)["group"]}
    provenance_blockers = []
    source_provenance = {}
    for area in recipe.areas:
        if area.method.model_id:
            model = store.model(area.method.model_id)
            used.update(model["fit_ids"]+model["calibration_ids"])
            groups.update(model["used_groups"])
            fit_groups = {store.read('images',i)['group'] for i in model['fit_ids']}
            calibration_groups = {store.read('images',i)['group'] for i in model['calibration_ids']}
            if (fit_groups | {store.read('images',recipe.reference_id)['group']}) & calibration_groups:
                provenance_blockers.append('Training/reference and calibration now share a capture group; rebuild the candidate with independent groups')
    for key in used:
        source = store.read('images',key)
        source_provenance[key] = {field:source.get(field) for field in ('label','group','confirmed_at','group_confirmed')}
        groups.add(source['group'])
        if not label_confirmed(source) or not group_confirmed(source):
            provenance_blockers.append('Reference/training/calibration sources need confirmed labels and explicit capture groups')
    rows = []
    for key in image_ids:
        image = store.read("images",key)
        result,_ = pipeline.run(store.image(key))
        exclusion = "Used as reference, training or calibration (including same capture group)" if key in used or image["group"] in groups else "Missing human-confirmed label" if not label_confirmed(image) else "Missing confirmed capture group" if not group_confirmed(image) else ""
        rows.append({"image_id":key,"name":image["name"],"label":image["label"],"group":image["group"],
                     "excluded":exclusion,"result":result})
    eligible_rows = [r for r in rows if not r["excluded"]]
    good = [r for r in eligible_rows if r["label"]=="GOOD"]
    ng = [r for r in eligible_rows if r["label"]=="NG"]
    counts = Counter(r["result"]["state"] for r in eligible_rows)
    rate = lambda count,total: count/total if total else None
    fa = sum(r["result"]["state"]=="PASS" for r in ng)
    fr = sum(r["result"]["state"]=="NG" for r in good)
    metrics = {"good_count":len(good),"ng_count":len(ng),"excluded_count":len(rows)-len(eligible_rows),
        "good_accepted":sum(r["result"]["state"]=="PASS" for r in good),
        "defects_rejected":sum(r["result"]["state"]=="NG" for r in ng),
        "false_accept_count":fa,"false_reject_count":fr,
        "false_accept_rate":rate(fa,len(ng)),"false_reject_rate":rate(fr,len(good)),
        "defect_recall":rate(sum(r['result']['state']=='NG' for r in ng),len(ng)),
        "ng_review_count":sum(r['result']['state']=='REVIEW' for r in ng),
        "ng_unable_count":sum(r['result']['state']=='UNABLE_TO_INSPECT' for r in ng),
        "review_rate":rate(counts["REVIEW"],len(eligible_rows)),
        "unable_rate":rate(counts["UNABLE_TO_INSPECT"],len(eligible_rows)),"counts":dict(counts),
        "mean_ms":float(np.mean([r["result"]["elapsed_ms"] for r in rows])),
        "p95_ms":float(np.percentile([r["result"]["elapsed_ms"] for r in rows],95))}
    limits = recipe.validation
    blockers = list(dict.fromkeys(provenance_blockers))
    if len(good)<limits.min_good:
        blockers.append(f"Need at least {limits.min_good} independent GOOD images")
    if len(ng)<limits.min_ng:
        blockers.append(f"Need at least {limits.min_ng} independent NG images; defect recall is unverified")
    for metric,limit in (("false_accept_rate",limits.max_false_accept),("false_reject_rate",limits.max_false_reject),
                         ("review_rate",limits.max_review),("unable_rate",limits.max_unable)):
        if metrics[metric] is None or metrics[metric] > limit:
            blockers.append(f"{metric} does not meet limit {limit:g}")
    if recipe.locator.method=="fixed":
        blockers.append("Continuous place-and-remove Production needs a part locator; fixed position supports image tests only")
    from .detection_metrics import evaluate
    object_metrics,object_blockers=evaluate(store,pipeline,rows,dataset_id)
    blockers.extend(object_blockers)
    report = {"id":"test_"+uuid.uuid4().hex,"engine_version":ENGINE_VERSION,"created_at":now(),"recipe_id":pipeline.saved["id"],
        "recipe_hash":pipeline.saved["hash"],"metrics":metrics,"rows":rows,"limits":limits.model_dump(),
        "eligible":not blockers,"blockers":blockers,"source_provenance":source_provenance,
        "purpose":"Independent validation; no automatic activation"}
    if object_metrics:report['object_metrics']=object_metrics
    report["report_hash"] = digest(report)
    atomic_json(store.path("tests",report["id"]),report)
    return report
