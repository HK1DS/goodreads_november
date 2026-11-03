@echo off
rem LightGCN-backbone experiment (k=30, TO), no LLM/API calls.
rem   Phase 1: seed 2020 for LightGCN (fixed adjacency), IKGR_lgcn_dyn_nointent, IKGR_lgcn_dyn
rem   Phase 2: seeds 2021, 2022 for the same three specs (seed 2020 results are reused)
rem Double-click to start. Ctrl+C once to stop; double-click again to resume
rem (finished model x seed results are kept; the interrupted one restarts).
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set IKGR_POP_RERANK_SPECS=IKGR_kgoff,BPR,LightGCN,IKGR_dyn,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn
set "PY=%USERPROFILE%\.venvs\goodreads-ikgr\Scripts\python.exe"
set SPECS=LightGCN,IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn
echo [launcher] phase1 start %DATE% %TIME%>> run_k30_TO\lgcn_launcher_status.txt
"%PY%" -u run_pipeline.py --k 30 --split TO --steps BCDER --seeds 2020 --specs %SPECS%
echo [launcher] phase1 exit=%ERRORLEVEL% %DATE% %TIME%>> run_k30_TO\lgcn_launcher_status.txt
if errorlevel 1 goto end
"%PY%" -u run_pipeline.py --k 30 --split TO --steps BCDER --seeds 2020,2021,2022 --specs %SPECS%
echo [launcher] phase2 exit=%ERRORLEVEL% %DATE% %TIME%>> run_k30_TO\lgcn_launcher_status.txt
:end
pause
