@echo off
rem embedding_size 64 re-run of the k=30 TO comparison (no LLM/API calls).
rem   Phase 1: seed 2020 for all seven specs
rem   Phase 2: seeds 2021, 2022 for the five core specs (seed 2020 results are reused)
rem Double-click to start. Ctrl+C once to stop; run again to resume
rem (finished model x seed results are kept; the interrupted one restarts).
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set IKGR_POP_RERANK_SPECS=IKGR_kgoff,BPR,LightGCN,IKGR_dyn,IKGR_dyn_nointent,IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn
set "PY=%USERPROFILE%\.venvs\goodreads-ikgr\Scripts\python.exe"
echo [launcher] emb64 phase1 start %DATE% %TIME%>> run_k30_TO\emb64_launcher_status.txt
"%PY%" -u run_pipeline.py --k 30 --split TO --steps BCDER --seeds 2020 --specs IKGR_lgcn_dyn_nointent,LightGCN,IKGR_lgcn_dyn,IKGR_dyn_nointent,BPR,IKGR_dyn,IKGR_kgoff
echo [launcher] emb64 phase1 exit=%ERRORLEVEL% %DATE% %TIME%>> run_k30_TO\emb64_launcher_status.txt
if errorlevel 1 goto end
"%PY%" -u run_pipeline.py --k 30 --split TO --steps BCDER --seeds 2020,2021,2022 --specs IKGR_lgcn_dyn_nointent,LightGCN,IKGR_lgcn_dyn,IKGR_dyn_nointent,BPR
echo [launcher] emb64 phase2 exit=%ERRORLEVEL% %DATE% %TIME%>> run_k30_TO\emb64_launcher_status.txt
:end
pause
