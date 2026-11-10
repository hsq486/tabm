# E3b: TabM with piecewise-linear embeddings (isolates the embedding confound).
#
# Datasets: california, churn, house
# Seeds:    0, 1, 2
#
# Frozen templates: `paper/configs/E3b-tabm-plr-<dataset>.toml`
# (k=8, 2x256, share_training_batches=true, official tuned lr/dropout/bins).

param(
    [string]$Python = "D:\Anaconda\envs\tabm\python.exe"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Model = Join-Path $Root "bin\model.py"
$ConfigsDir = Join-Path $Root "configs"
$ResultsDir = Join-Path $Root "results\E3b"
$CsvPath = Join-Path $ResultsDir "runs.csv"

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

$Total = $Datasets.Count * $Seeds.Count
$Index = 0
$Failures = 0

foreach ($Dataset in $Datasets) {
    $Template = Join-Path $ConfigsDir "E3b-tabm-plr-$Dataset.toml"
    if (-not (Test-Path $Template)) {
        Write-Warning "Missing frozen config: $Template"
        $Failures += $Seeds.Count
        continue
    }

    $RunDir = Join-Path $Root "exp\reproduce\E3b\$Dataset"
    New-Item -ItemType Directory -Force $RunDir | Out-Null

    foreach ($Seed in $Seeds) {
        $Index += 1
        $ConfigPath = Join-Path $RunDir "$Seed.toml"
        $LogPath = Join-Path $RunDir "$Seed.log"
        $ReportPath = Join-Path $RunDir "$Seed\report.json"

        $ConfigText = Get-Content -Path $Template -Raw
        $ConfigText = $ConfigText -replace "(?m)^seed\s*=\s*\d+", "seed = $Seed"
        # tomli rejects a UTF-8 BOM; PowerShell 5 Set-Content -Encoding UTF8 adds one.
        [System.IO.File]::WriteAllText(
            $ConfigPath,
            $ConfigText,
            (New-Object System.Text.UTF8Encoding($false))
        )

        Write-Host "[$Index/$Total] tabm-piecewiselinear / $Dataset / seed=$Seed"
        $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        # With "Stop", PowerShell treats the first stderr write (tqdm progress)
        # of a native command as a terminating error and kills the run.
        $SavedErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Python $Model $ConfigPath --force 2>&1 |
                ForEach-Object {
                    if ($_ -is [System.Management.Automation.ErrorRecord]) {
                        $_.ToString()
                    }
                    else {
                        $_
                    }
                } |
                Out-File -FilePath $LogPath -Encoding utf8
            $ExitCode = $LASTEXITCODE
        }
        catch {
            $ExitCode = -1
            $_ | Out-String | Add-Content -Path $LogPath
        }
        finally {
            $ErrorActionPreference = $SavedErrorAction
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
            method       = "tabm-piecewiselinear"
            dataset      = $Dataset
            seed         = $Seed
            exit_code    = $ExitCode
            duration_sec = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 1)
            status       = $Status
            report       = $ReportPath
        } | Export-Csv -Path $CsvPath -Append -NoTypeInformation -Encoding UTF8
    }
}

Write-Host "E3b finished: $($Total - $Failures)/$Total runs ok; ledger: $CsvPath"
if ($Failures -gt 0) {
    exit 1
}
