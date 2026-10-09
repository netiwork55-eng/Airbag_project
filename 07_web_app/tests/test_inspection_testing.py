import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection_testing import summarize_test

def row(expected,decision,**kw):
    return dict(expected=expected,decision=decision,overlap_check_available=True,**kw)

def test_no_ng_data_is_unknown_not_perfect_recall():
    data=summarize_test([row("OK","OK"),row("OK","REVIEW")])
    assert data["good_acceptance"]==.5
    assert data["good_unresolved"]==1
    assert data["defect_detection"] is None

def test_errors_and_review_do_not_disappear_from_denominator():
    data=summarize_test([row("NG",x) for x in ["NG","OK","REVIEW","NO_PART","ERROR"]])
    assert data["defect_detection"]==.2 and data["defect_miss"]==.2
    assert data["bad_unresolved"]==3

def test_development_images_are_excluded_from_reported_rates():
    data=summarize_test([row("NG","NG",development_image=True),row("UNKNOWN","OK")])
    assert data["defect_detection"] is None
    assert data["development_images"]==1 and data["unknown_labels"]==1


def test_compare_uses_each_model_once_and_excludes_training_overlap(tmp_path,monkeypatch):
    import json
    import numpy as np
    import app
    model=tmp_path/'model.npz';model.write_bytes(b'model')
    model.with_suffix('.training.json').write_text(json.dumps({'fit':[{'sha256':'shared'}],'calibration':[],'validation':[]}))
    monkeypatch.setattr(app,'WEB_APP_DIR',tmp_path)
    monkeypatch.setattr(app,'job_change_blocked',lambda:None)
    calls=[]
    def context(value):
        calls.append(value)
        return ({'id':value,'name':value,'files':{'model':{'path':str(model),'sha256':'model-hash'}}},)
    monkeypatch.setattr(app,'load_job_test_context',context)
    monkeypatch.setattr(app,'run_job_test_image',lambda context,image,preview: {'decision':'OK','test_ms':1.})
    samples=[{'name':'good.png','expected':'OK','sha256':'shared','image':np.zeros((2,2,3))},
             {'name':'other.png','expected':'UNKNOWN','sha256':'other','image':np.ones((2,2,3))}]
    response=app.compare_job_images(['a','b'],samples)
    data=json.loads(response.body)
    assert response.status_code==200 and calls==['a','b']
    for model in data['models']:
        assert model['summary']['development_images']==1
        assert model['summary']['good_acceptance'] is None
    assert (tmp_path/'logs/model_tests'/(data['id']+'.json')).is_file()


def test_report_path_cannot_escape_folder():
    import app
    assert app.api_test_report('../project_config').status_code==400
    assert app.api_part_image('../project_config').status_code==400
