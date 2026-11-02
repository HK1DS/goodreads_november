# Goodreads train-only revalidation: one-shot runner for Windows (PowerShell 5.1+).
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_goodreads_revalidation.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_goodreads_revalidation.ps1 -K 30 -Split TO -Seeds 2020
#
# Order: (0) Python env: -Python, else repo .venv, else create/reuse %USERPROFILE%\.venvs\goodreads-ikgr
#        (1) download raw Goodreads files if missing or Git-LFS pointers
#        (2) free data prep: preprocess -> k-core -> temporal .inter -> train-only profiles (reused if present)
#        (3) LLM cost estimate, then ASK before any paid call
#        (4) step1 + step2 (paid) -> KG packs -> early-stopped multi-seed eval -> report
# Everything is resumable: LLM caches, finished seeds and existing files are reused.
param(
    [int]$K = 30,
    [ValidateSet("TO", "TO_GLOBAL")][string]$Split = "TO",
    [string]$Seeds = "2020,2021,2022",
    [string]$Specs = "IKGR_kgoff,BPR,LightGCN,IKGR_full_hetero,IKGR_dyn_nointent,IKGR_dyn",
    [int]$MaxEpochs = 40,
    [string]$PopGrid = "0.25,0.5,1.0",
    [string]$ProfileStyle = "structured",
    [switch]$DumpValid,
    [switch]$SkipLLM,
    [switch]$SkipDownload,
    [switch]$Yes,
    [string]$Python = "",
    [string]$VenvDir = "$env:USERPROFILE\.venvs\goodreads-ikgr",
    [string]$BasePython = ""
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = "utf-8"

# ---------------------------------------------------------------- 0) Python environment
if (-not $Python) {
    if (Test-Path ".venv\Scripts\python.exe") {
        $Python = ".venv\Scripts\python.exe"
    } else {
        Write-Host "[env] no -Python given and no repo .venv -> preparing $VenvDir (first run: ~10-20 min, ~5 GB)"
        & (Join-Path $PSScriptRoot "setup_venv.ps1") -VenvDir $VenvDir -BasePython $BasePython
        if ($LASTEXITCODE -ne 0) { Write-Host "[env] environment setup failed (see messages above)."; exit 1 }
        if ($IsLinux -or $IsMacOS) { $Python = Join-Path $VenvDir "bin/python" } else { $Python = Join-Path $VenvDir "Scripts\python.exe" }
    }
}
Write-Host "[run] repo=$(Get-Location) python=$Python k=$K split=$Split seeds=$Seeds"

# Fail fast (before any paid call) if the training stack is not importable.
& $Python -c "import ssl, torch, recbole, sentence_transformers, sklearn, scipy, yaml, requests; print('[env] torch', torch.__version__, '| recbole', recbole.__version__, '| cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[env] $Python cannot import the pipeline packages. Run scripts\setup_venv.ps1 or pass -Python <env python.exe>."
    exit 1
}
$cuda = & $Python -c "import torch; print(int(torch.cuda.is_available()))"
if ("$cuda".Trim() -ne "1") { Write-Host "[env] WARNING: CUDA not available -> training will run on CPU (very slow)." }

function Invoke-Py([string[]]$PyArgs) {
    Write-Host "`n> $Python $($PyArgs -join ' ')"
    & $Python @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "python exited with $LASTEXITCODE : $($PyArgs -join ' ')" }
}

# ---------------------------------------------------------------- 1) raw data
$bases = @(
    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/",
    "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/"
)
$files = [ordered]@{
    "goodreads_interactions_children.json.gz" = @("byGenre/goodreads_interactions_children.json.gz", $true)
    "goodreads_books_children.json.gz"        = @("byGenre/goodreads_books_children.json.gz", $true)
    "goodreads_book_authors.json.gz"          = @("goodreads_book_authors.json.gz", $false)
}
if (-not $SkipDownload) {
    foreach ($name in $files.Keys) {
        $rel = $files[$name][0]; $required = $files[$name][1]
        $ok = (Test-Path $name) -and ((Get-Item $name).Length -gt 1MB)
        if ($ok) { Write-Host "[data] $name present"; continue }
        if (Test-Path $name) { Write-Host "[data] $name is a Git-LFS pointer ($((Get-Item $name).Length) bytes) -> downloading" }
        $done = $false
        foreach ($b in $bases) {
            try {
                Write-Host "[data] GET $b$rel"
                Invoke-WebRequest -Uri ($b + $rel) -OutFile "$name.part" -UseBasicParsing -TimeoutSec 3600
                if ((Get-Item "$name.part").Length -gt 1MB) { Move-Item -Force "$name.part" $name; $done = $true; break }
            } catch { Write-Host "[data]   failed: $($_.Exception.Message)" }
        }
        if (Test-Path "$name.part") { Remove-Item "$name.part" -Force }
        if (-not $done) {
            if ($required) {
                Write-Host "[data] Could not download $name. Get it from https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/UCSD%20Book%20Graph.html (Children genre) and put it in the repo root, then re-run."
                exit 1
            } else {
                Write-Host "[data] optional $name unavailable -> profiles will use titles without author names."
            }
        }
    }
}

# ---------------------------------------------------------------- 2) free prep + 3) estimate
try {
    Invoke-Py @("-u", "run_pipeline.py", "--k", "$K", "--split", $Split, "--steps", "0ATPS",
                "--profile_style", $ProfileStyle)
} catch { Write-Host $_; exit 1 }

$runDir = "run_k$($K)_$Split"
$est = Get-Content (Join-Path $runDir "llm_cost_estimate.json") -Raw | ConvertFrom-Json
$calls = [int]$est.total.calls
Write-Host ""
Write-Host ("[cost] pending LLM calls: {0:N0} (step1 {1:N0} + step2 {2:N0})" -f $calls, $est.step1.calls, $est.step2.calls)
Write-Host ("[cost] list-price estimate ~`${0}; observed-rate estimate ~`${1}-{2}" -f `
    $est.total.usd_list_price, $est.total.usd_observed_range[0], $est.total.usd_observed_range[1])

if ($SkipLLM) { Write-Host "[run] -SkipLLM given: stopping after free prep."; exit 0 }
if ($calls -gt 0 -and -not $Yes) {
    $ans = Read-Host "Spend this on the Luxia API now? [y/N]"
    if ($ans -notin @("y", "Y", "yes")) { Write-Host "[run] stopped before any paid call."; exit 0 }
}

# ---------------------------------------------------------------- 4) LLM -> KG -> eval -> report
$pyArgs = @("-u", "run_pipeline.py", "--k", "$K", "--split", $Split, "--steps", "BCDER", "--yes",
            "--epochs", "$MaxEpochs", "--seeds", $Seeds, "--specs", $Specs, "--pop_grid", $PopGrid)
if ($DumpValid) { $pyArgs += "--dump_valid" }
try { Invoke-Py $pyArgs } catch { Write-Host $_; exit 1 }

Write-Host "`n[done] results: $runDir\slice_eval_$($Split)_result.json"
Write-Host "[done] report : $runDir\report_$Split.md"
