@echo off
rem 64-dim LightGCN seed 2021 only (seed 2020 result is reused/skipped), then the paired report. No LLM/API calls.
rem Ctrl+C once to stop; run again to resume (the interrupted model restarts from epoch 0).
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set IKGR_POP_RERANK_SPECS=IKGR_kgoff,BPR,LightGCN,IKGR_dyn,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn
set "PY=%USERPROFILE%\.venvs\goodreads-ikgr\Scripts\python.exe"
echo [launcher] emb64 lightgcn-s2021 start %DATE% %TIME%>> run_k30_TO\emb64_launcher_status.txt
"%PY%" -u run_pipeline.py --k 30 --split TO --steps BCDER --seeds 2020,2021 --specs LightGCN
echo [launcher] emb64 lightgcn-s2021 exit=%ERRORLEVEL% %DATE% %TIME%>> run_k30_TO\emb64_launcher_status.txt
pause
