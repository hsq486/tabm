# Run the E2 baselines sequentially and record every run.
#
# Baselines:
#   - mlp                  (official tuned configuration)
#   - mlp-piecewiselinear  (official tuned configuration)
#
# Datasets: california, churn, house
# Seeds:    0, 1, 2
#
# Each run derives its config from the official `0-evaluation/0.toml`, changes
# only the seed, and writes outputs under `exp/reproduce/E2-baseline/`.
# Raw logs and reports stay local (ignored by Git); `results/E2/runs.csv`
# is the small, reviewable execution ledger.

param(
    [string]$Python = "D:\Anaconda\envs\tabm\python.exe"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Model = Join-Path $Root "bin\model.py"
$ResultsDir = Join-Path $Root "results\E2"
$CsvPath = Join-Path $ResultsDir "runs.csv"

$Methods = @("mlp", "mlp-piecewiselinear")
$Datasets = @("california", "churn", "house")
$Seeds = @(0, 1, 2)

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path $Model)) {
    throw "Model entry point not found: $Model"
}

New-Item -ItemType Directory -Force $ResultsDir | Out-Null
"timestamp,commit,method,dataset,seed,exit_code,duration_sec,status,report" |
    Set-Content -Path $CsvPath -Encoding UTF8

$Commit = (git -C $Root rev-parse --short HEAD).Trim()
$env:OMP_NUM_THREADS = "14"
$env:MKL_NUM_THREADS = "14"

$Total = $Methods.Count * $Datasets.Count * $Seeds.Count
$Index = 0
$Failures = 0

foreach ($Method in $Methods) {
    foreach ($Dataset in $Datasets) {
        $OfficialConfig = Join-Path $Root "exp\$Method\$Dataset\0-evaluation\0.toml"
        if (-not (Test-Path $OfficialConfig)) {
            Write-Warning "Missing official config: $OfficialConfig"
            $Failures += 1
            continue
        }

        $RunDir = Join-Path $Root "exp\reproduce\E2-baseline\$Method\$Dataset"
        New-Item -ItemType Directory -Force $RunDir | Out-Null

        foreach ($Seed in $Seeds) {
            $Index += 1
            $ConfigPath = Join-Path $RunDir "$Seed.toml"
            $LogPath = Join-Path $RunDir "$Seed.log"
            $ReportPath = Join-Path $RunDir "$Seed\report.json"

            $ConfigText = Get-Content -Path $OfficialConfig -Raw
            $ConfigText = $ConfigText -replace "(?m)^seed\s*=\s*\d+", "seed = $Seed"
            # tomli rejects a UTF-8 BOM; PowerShell 5 Set-Content -Encoding UTF8 adds one.
            [System.IO.File]::WriteAllText(
                $ConfigPath,
                $ConfigText,
                (New-Object System.Text.UTF8Encoding($false))
            )

            Write-Host "[$Index/$Total] $Method / $Dataset / seed=$Seed"
            $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            try {
                & $Python $Model $ConfigPath --force *> $LogPath
                $ExitCode = $LASTEXITCODE
            }
            catch {
                $ExitCode = -1
                $_ | Out-String | Add-Content -Path $LogPath
            }
            $Stopwatch.Stop()

            $HasDone = Test-Path (Join-Path $RunDir "$Seed\DONE")
            $Status = if ($ExitCode -eq 0 -and $HasDone) { "ok" } else { "failed" }
            if ($Status -ne "ok") {
                $Failures += 1
            }

            [pscustomobject]@{
                timestamp    = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
                commit       = $Commit
                method       = $Method
                dataset      = $Dataset
                seed         = $Seed
                exit_code    = $ExitCode
                duration_sec = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 1)
                status       = $Status
                report       = $ReportPath
            } | Export-Csv -Path $CsvPath -Append -NoTypeInformation -Encoding UTF8
        }
    }
}

Write-Host "E2 finished: $($Total - $Failures)/$Total runs ok; ledger: $CsvPath"
if ($Failures -gt 0) {
    exit 1
}
