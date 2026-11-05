@echo off
rem 64-dim seed 2021 for IKGR_lgcn_dyn_nointent, then LightGCN (seed 2020 results are reused/skipped),
rem then the paired report over the available seeds. No LLM/API calls.
rem Ctrl+C once to stop; run again to resume (finished model x seed results are kept).
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set IKGR_POP_RERANK_SPECS=IKGR_kgoff,BPR,LightGCN,IKGR_dyn,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn
set "PY=%USERPROFILE%\.venvs\goodreads-ikgr\Scripts\python.exe"
echo [launcher] emb64 seed2021 start %DATE% %TIME%>> run_k30_TO\emb64_launcher_status.txt
"%PY%" -u run_pipeline.py --k 30 --split TO --steps BCDER --seeds 2020,2021 --specs IKGR_lgcn_dyn_nointent,LightGCN
echo [launcher] emb64 seed2021 exit=%ERRORLEVEL% %DATE% %TIME%>> run_k30_TO\emb64_launcher_status.txt
pause
