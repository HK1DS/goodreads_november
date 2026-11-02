@echo off
rem Seeds 2020-2022 for IKGR_dyn_nointent, IKGR_dyn, LightGCN, BPR (k=30, TO).
rem Double-click to start. Ctrl+C once to stop; double-click again to resume (finished model x seed results are kept).
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set IKGR_POP_RERANK_SPECS=IKGR_kgoff,BPR,LightGCN,IKGR_dyn,IKGR_dyn_nointent
set "PY=%USERPROFILE%\.venvs\goodreads-ikgr\Scripts\python.exe"
echo [launcher] start %DATE% %TIME%>> run_k30_TO\seeds_launcher_status.txt
"%PY%" -u run_pipeline.py --k 30 --split TO --steps BCDER --seeds 2020,2021,2022 --specs IKGR_dyn_nointent,IKGR_dyn,LightGCN,BPR
echo [launcher] exit=%ERRORLEVEL% %DATE% %TIME%>> run_k30_TO\seeds_launcher_status.txt
pause
