"""Prepare local deliverables only; no email, uploads or GitHub mutations."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import gzip

import numpy as np
import pandas as pd

W=Path(__file__).resolve().parent
REPO=W/'repo'
OUT=W.parent/'outputs'
BUILD=W/'delivery_files'
BUILD.mkdir(exist_ok=True)
sys.path.insert(0,str(REPO))
import paired_compare as pc

D=REPO/'run_k30_TO/todo_5_8'
R=json.loads((D/'slice_eval_TO_result.json').read_text())
SEEDS=[2020,2021,2022]
BASE=['BPR','IKGR_kgoff','IKGR_dyn_nointent','IKGR_dyn','IKGR_dyn_filter5','IKGR_dyn_filter10','IKGR_dyn_filter20']
RATES=['IKGR_dyn','IKGR_dyn_filter5','IKGR_dyn_filter10','IKGR_dyn_filter20']
manifest=json.loads((OUT/'formal-run-manifest.json').read_text())
for name,h in manifest['files_sha256'].items():
    assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==h, name
assert json.loads((D/'control/status.json').read_text())['state']=='completed'

def write_csv(name,rows):
    with (BUILD/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# Validate every gzip stream and population before publishing raw artifacts.
raw=[]
for spec in BASE:
    for seed in SEEDS:
        for suffix in ['', '_pop0p25','_pop0p5','_pop1p0']:
            name=spec+suffix
            assert str(seed) in R[name]['seeds']
            for phase in ['valid','test']:
                path=D/'per_user_TO'/f'{name}__s{seed}__{phase}.csv.gz'
                lines=0
                with gzip.open(path,'rb') as f:
                    for block in iter(lambda:f.read(1<<20),b''): lines+=block.count(b'\n')
                assert lines==62143,(path,lines)
                raw.append({'file':path.name,'bytes':path.stat().st_size,'rows':lines-1,
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
print('Verified 168 gzip files: full CRC and 62,142 rows each',flush=True)

validation={}
for spec in RATES:
    validation[spec]=[]
    for seed in SEEDS:
        f=D/'per_user_TO'/f'{spec}__s{seed}__valid.csv.gz'
        v=pd.read_csv(f,usecols=['ndcg@10'])['ndcg@10'].mean()
        validation[spec].append(float(v))
selected=max(RATES,key=lambda n:(np.mean(validation[n]),-RATES.index(n)))

seed_rows=[]; aggregates=[]; timings=[]
for name,entry in R.items():
    for seed in SEEDS:
        x=entry['seeds'][str(seed)]
        row={'result':name,'seed':seed,**x['overall'],
             'tail_recall@10':x['long_tail']['recall@10'],'tail_recall@30':x['long_tail']['recall@30'],
             'coverage@10':x['coverage@10'],'novelty_bits':x['novelty_bits'],
             'best_epoch_zero_based':x['fit']['best_epoch'],'epochs_run':x['fit']['last_epoch']+1,
             'fit_seconds_including_validation':x['train_sec']}
        seed_rows.append(row)
        if name in BASE:
            timings.append({k:row[k] for k in ['result','seed','epochs_run','best_epoch_zero_based','fit_seconds_including_validation']})
    a={'result':name,'seed_count':3}
    for metric in ['ndcg@10','recall@10','tail_recall@10','coverage@10']:
        values=[q[metric] for q in seed_rows if q['result']==name]
        a[metric+'_mean']=float(np.mean(values));a[metric+'_std_ddof0']=float(np.std(values,ddof=0))
    a['valid_ndcg@10_mean']=float(np.mean(validation[name])) if name in validation else ''
    aggregates.append(a)
write_csv('all_results_by_seed.csv',seed_rows)
write_csv('all_results_mean_std.csv',aggregates)
write_csv('fit_times.csv',timings)
write_csv('validation_selection.csv',[{'result':n,**{str(s):validation[n][i] for i,s in enumerate(SEEDS)},
           'mean':float(np.mean(validation[n])),'selected':n==selected} for n in RATES])

# Existing repository methodology: average each user's repeated seeds BEFORE bootstrap.
cache={}
def users(name,metric):
    key=(name,metric)
    if key not in cache:
        frame,count=pc.load_user_means(str(D/'per_user_TO'),name,metric,'test',set(SEEDS))
        assert count==3 and frame.index.is_unique
        cache[key]=frame
    return cache[key]
pairs=[(n,'IKGR_dyn') for n in RATES[1:]]+[(n,'IKGR_dyn_nointent') for n in RATES]
pairs += [(selected,'BPR'),(selected,'IKGR_kgoff')]
stats=[]
for a,b in pairs:
    for metric in ['ndcg@10','tail_recall@10']:
        ua,ub=users(a,metric),users(b,metric)
        assert set(ua.index)==set(ub.index)
        p=pc.paired(ua,ub,n_boot=10000,seed=0)
        stats.append({'a':a,'b':b,'metric':metric,**{k:v for k,v in p.items() if k!='joined'}})
    print('Paired analysis:',a,'vs',b,flush=True)
(BUILD/'paired_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
write_csv('paired_statistics.csv',[{**{k:v for k,v in r.items() if k!='ci95'},'ci95_low':r['ci95'][0],'ci95_high':r['ci95'][1]} for r in stats])

summary=['Goodreads TODO 5·8 정식 실험 결과','',
 '완료: 2026-11-10 08:25:24 KST. 21회 학습, 인기도 재랭킹 포함 84개 모델×seed 결과, 사용자별 파일 168개.',
 '용도: 이메일 등으로 전달할 로컬 파일. 메일 발송·업로드·PR 준비/생성은 수행하지 않음.','',
 '핵심 비교 (test, 3 seed 평균 ± 모집단 표준편차 ddof=0; 원본 JSON의 seed별 반올림 값 기준)',
 '모델 | NDCG@10 | tail Recall@10 | Coverage@10 | validation NDCG@10']
for name in BASE:
    a=next(q for q in aggregates if q['result']==name)
    valid=f"{a['valid_ndcg@10_mean']:.6f}" if a['valid_ndcg@10_mean']!='' else '-'
    summary.append(f"{name} | {a['ndcg@10_mean']:.5f} ± {a['ndcg@10_std_ddof0']:.5f} | "
                   f"{a['tail_recall@10_mean']:.5f} | {a['coverage@10_mean']:.5f} | {valid}")
summary+=['',f'대표 제거율 선택: {selected}. 3 seed validation NDCG@10 평균으로만 선택; 동률이면 낮은 제거율.',
 'Test 결과는 선택에 사용하지 않음. 사용자별 원본으로 계산한 validation 평균과 반올림된 fit 점수의 표시값은 다를 수 있음.',
 '효과의 크기와 불확실성은 paired_statistics.csv/json 참조. p값은 다중 비교 보정 전이며 탐색적 비교로 해석.','',
 '원래 저장소 요구사항과 이번 결정',
 '- TODO 5: BPR/IKGR_kgoff/IKGR_dyn_nointent × seeds 2020,2021,2022 (9회).',
 '- TODO 8 원문: 연결 아이템 수 상위 5/10/20% intent 제거. 이번 결정: MF 기반 IKGR_dyn의 0/5/10/20% × 3 seed (12회).',
 '- train 고유 아이템 연결 수; positive degree intent만 분모; degree 내림차순/ID 오름차순; floor 비율.',
 '- 사용자/아이템 양쪽 연결 제거, ID/임베딩/남은 순서 보존, 기존 cap32 적용 전 필터. frozen intent 유지.',
 '- 제거 대상 114,794개; 제거 수 0/5,739/11,479/22,958개. 제거 목록과 지문 동봉.',
 '- 기존 프로토콜 유지: k30, 사용자별 TO 80/10/10, train-only 프로필·recency, 64d, 최대40epoch, lr0.001, reg1e-6.',
 '- 기존 코드의 eval_step=2, stopping_step=3 유지. RecBole은 미개선 횟수가 3을 초과할 때 종료하고 동점은 갱신하므로, 문서의 3회 연속 미개선 표현과 차이가 있음.',
 '- validation NDCG@10 최고 체크포인트로 test. 모든 비교를 같은 RTX2070SUPER, 동일 seed에서 수행.',
 '- 이번 설정: train batch2048 유지, 평가128명/배치. validation/test 각각62,142명 전체 및 전체 아이템 평가.',
 '- 1명/128명 실측에서 일부 근접 동률 순위 차이가 있었음. 배치128을 모든 정식 조건에 동일 적용.',
 '- 인기도 페널티 0.25/0.5/1.0은 기존 방식대로 추가 학습 없이 모든 조건에 평가.',
 '- 통계는 기존 paired_compare.py 사용: 사용자별 seed 평균 후 bootstrap10,000회/95% CI, Wilcoxon(zsplit), bootstrap seed0.',
 '- 결과 해석 범위는 MF 백본이며 LightGCN으로 일반화하지 않음. 레거시/512d/임시 벤치마크 지표와 섞지 않음.','',
 '실행 이력과 시간',
 '- 최초 시작 02:11:30, 완료08:25:24 KST: 경과 약6시간14분(재시작 공백 및 손실 포함).',
 '- 05:26경 필터5% seed2020의 12번째 epoch 기록 이후 프로세스 소실. 재부팅/정상 중지/Python 오류 기록 없이 원인 미확정.',
 '- 사용자 요청 일시중지가 아니어서 재개 스냅샷이 없었음. 완료12회는 그대로 재사용하고 해당1회만 epoch0부터 재실행.',
 '- 05:30:51 작업 제어 그룹에서 분리된 프로세스로 재실행. 이후 정상 exit0, GPU 반환.',
 '- fit_times.csv는 학습+주기 검증+최고 모델 저장 시간이며, 데이터 준비·최종 평가·내보내기·중단된 미완료 작업 시간은 제외.',
 '- 요청 시 저장/재개 기능은 CPU/GPU 합성 데이터에서 연속 실행과 상태·난수·결과 일치를 검증했으나, 이번 정식 실행에서는 요청 중지 기능을 쓰지 않음.',
 '- 단위/통합 검사 기록 동봉. 일반 패키지 경고 외 최종 정식 실행 오류 없음.','',
 '파일 구성',
 '- Goodreads_TODO5_8_results.zip: 이 요약, 지표/통계/시간 CSV, 원본 결과 JSON, 필터 목록, 선택/프로토콜, 변경 패치·소스, 환경·실행 해시, 로그·검증 기록.',
 '- Goodreads_TODO5_8_per_user.zip: 원본168개 gzip CSV 및 행 수/SHA256 명세. 대용량이므로 일반 메일 첨부보다 별도 전달에 적합.',
 '- 학습 체크포인트와 복원한 대규모 입력 데이터는 전달 묶음에서 제외. 입력은 원본 architect/data_bundle에서 복원 가능.',
 '- reproduce.txt: 원본 기준 커밋, 코드 적용 및 같은 전용 환경 설정 방법.',
 '- 파일은 준비만 했으며 이메일 초안·메일 발송·PR·외부 업로드를 만들지 않았음.']
(BUILD/'summary.txt').write_text('\n'.join(summary)+'\n',encoding='utf-8-sig')
(BUILD/'raw_manifest.json').write_text(json.dumps(raw,indent=2),encoding='utf-8')
(BUILD/'reproduce.txt').write_text('Base: '+manifest['base_commit']+'\n'
 'Repository: https://github.com/HK1DS/goodreads_november\n'
 '1. Clone and checkout the base commit.\n2. git apply prepared-experiment.patch\n'
 '3. Create a dedicated Python 3.11.18 environment; use environment-requirements.txt (torch2.3.0+cu121).\n'
 '4. Restore data_bundle with python scripts/unpack_data_bundle.py.\n'
 '5. python scripts/run_todo_5_8.py --check\n6. python scripts/run_todo_5_8.py --run --phase all\n'
 'Use a clean output directory; delivered results are under results/ for review. Original paths are recorded in manifests.\n'
 'Extra analysis: work/prepare_delivery_files.py; paired_compare.py; fixed seeds2020/2021/2022, bootstrap10000 seed0.\n',encoding='utf-8')

core=OUT/'Goodreads_TODO5_8_results.zip'
files={p.name:p for p in BUILD.iterdir() if p.is_file()}
for name in ['prepared-experiment.patch','environment-requirements.txt','formal-run-manifest.json','formal-preflight.json',
             'experiment-decisions.txt','pause-resume-guide.txt','pause-resume-tests.txt','settings-tests.txt','settings-smoke.txt']:
    files['evidence/'+name]=OUT/name
for name in ['formal.stdout.log','formal.stderr.log','pause_tests_final.log','pause_launcher_test.log','pause_smoke_final.log']:
    files['logs/'+name]=W/name
launch=json.loads((W/'formal-launch.json').read_text(encoding='utf-8-sig'))
for key in ['stdout','stderr']:
    path=Path(launch[key]);files['logs/'+path.name]=path
files['evidence/latest_launch.json']=W/'formal-launch.json'
files['evidence/final_status.json']=D/'control/status.json'
for path in D.glob('*.json'):files['results/'+path.name]=path
for path in ['TODO.md','ARCHITECTURE.md','paired_compare.py','requirements-pipeline.txt',*manifest['files_sha256'].keys(),
             'tests/test_pause_resume.py','tests/pause_worker.py','tests/pause_queue_worker.py','tests/test_experiment_setup.py',
             'tests/test_lgcn_backbone.py','tests/smoke_e2e.py']:
    files['repository/'+path]=REPO/path
files['analysis/prepare_delivery_files.py']=Path(__file__)
digests={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
with zipfile.ZipFile(core,'w',zipfile.ZIP_DEFLATED) as z:
    for name,path in files.items():z.write(path,name)
    z.writestr('sha256.json',json.dumps(digests,indent=2))
rawzip=OUT/'Goodreads_TODO5_8_per_user.zip'
with zipfile.ZipFile(rawzip,'w',zipfile.ZIP_STORED) as z:
    for row in raw:z.write(D/'per_user_TO'/row['file'],'per_user_TO/'+row['file'])
    z.write(BUILD/'raw_manifest.json','raw_manifest.json')
for archive in [core,rawzip]:
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
shutil.copyfile(BUILD/'summary.txt',OUT/'Goodreads_TODO5_8_summary.txt')
(OUT/'Goodreads_TODO5_8_SHA256.txt').write_text('\n'.join(
    hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name for p in [core,rawzip,OUT/'Goodreads_TODO5_8_summary.txt'])+'\n',encoding='utf-8')
print(json.dumps({'selected_by_validation':selected,'files':len(files),'core_bytes':core.stat().st_size,
                  'raw_bytes':rawzip.stat().st_size}),flush=True)
