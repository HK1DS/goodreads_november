@echo off
rem 64-dim IKGR_lgcn_dyn (LightGCN backbone + metadata KG + recency + LLM intent), seed 2020 only,
rem then the paired report (incl. IKGR_lgcn_dyn vs IKGR_lgcn_dyn_nointent). No LLM/API calls.
rem Double-click to start. Ctrl+C once to stop; run again to resume (the interrupted model restarts from epoch 0).
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set IKGR_POP_RERANK_SPECS=IKGR_kgoff,BPR,LightGCN,IKGR_dyn,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn
set "PY=%USERPROFILE%\.venvs\goodreads-ikgr\Scripts\python.exe"
echo [launcher] emb64 lgcn_dyn-s2020 start %DATE% %TIME%>> run_k30_TO\emb64_launcher_status.txt
"%PY%" -u run_pipeline.py --k 30 --split TO --steps BCDER --seeds 2020 --specs IKGR_lgcn_dyn
echo [launcher] emb64 lgcn_dyn-s2020 exit=%ERRORLEVEL% %DATE% %TIME%>> run_k30_TO\emb64_launcher_status.txt
pause
