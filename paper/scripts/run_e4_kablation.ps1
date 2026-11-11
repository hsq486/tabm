# E4: integration-size ablation — TabM k ∈ {1,2,4,8,16,32}
#
# Datasets: california, churn (回归 + 分类，覆盖两类任务)
# Seeds:    0 only (单 seed 即可见趋势；论文 Figure 7 也是单 seed)
#
# Frozen templates: `paper/configs/E4-tabm-<dataset>.toml`
# (只替换 `k`，其余超参与 E3 主配置完全一致，以隔离集成规模效应)

param(
    [string]$Python = "D:\Anaconda\envs\tabm\python.exe"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Model = Join-Path $Root "bin\model.py"
$ConfigsDir = Join-Path $Root "configs"
$ResultsDir = Join-Path $Root "results\E4"
$CsvPath = Join-Path $ResultsDir "runs.csv"

$Datasets = @("california", "churn")
$Ks = @(1, 2, 4, 8, 16, 32)

if (-not (Test-Path $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path $Model)) { throw "model.py not found: $Model" }

New-Item -ItemType Directory -Force $ResultsDir | Out-Null
"timestamp,commit,dataset,k,exit_code,duration_sec,status,report" |
    Set-Content -Path $CsvPath -Encoding UTF8

$Commit = (git -C $Root rev-parse --short HEAD).Trim()
$env:OMP_NUM_THREADS = "14"
$env:MKL_NUM_THREADS = "14"

$enc = New-Object System.Text.UTF8Encoding($false)
$Total = $Datasets.Count * $Ks.Count
$Index = 0
$Failures = 0

foreach ($Dataset in $Datasets) {
    $Template = Join-Path $ConfigsDir "E4-tabm-$Dataset.toml"
    if (-not (Test-Path $Template)) { throw "Missing template: $Template" }
    $TemplateText = Get-Content -Path $Template -Raw

    foreach ($K in $Ks) {
        $Index += 1
        $RunDir = Join-Path $Root "exp\reproduce\E4\$Dataset\k=$K"
        New-Item -ItemType Directory -Force $RunDir | Out-Null

        $ConfigPath = Join-Path $RunDir "0.toml"
        $LogPath = Join-Path $RunDir "0.log"
        $ReportPath = Join-Path $RunDir "0\report.json"

        $ConfigText = $TemplateText -replace '(?m)^k\s*=\s*\d+', "k = $K"
        [System.IO.File]::WriteAllText($ConfigPath, $ConfigText, $enc)

        Write-Host "[$Index/$Total] TabM / $Dataset / k=$K"
        $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $SavedAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Python $Model $ConfigPath --force 2>&1 |
                ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ } } |
                Out-File -FilePath $LogPath -Encoding utf8
            $ExitCode = $LASTEXITCODE
        }
        catch {
            $ExitCode = -1
            $_ | Out-String | Add-Content -Path $LogPath
        }
        finally {
            $ErrorActionPreference = $SavedAction
        }
        $Stopwatch.Stop()

        $HasDone = Test-Path (Join-Path $RunDir "0\DONE")
        $Status = if ($ExitCode -eq 0 -and $HasDone) { "ok" } else { "failed" }
        if ($Status -ne "ok") { $Failures += 1 }

        [pscustomobject]@{
            timestamp    = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
            commit       = $Commit
            dataset      = $Dataset
            k            = $K
            exit_code    = $ExitCode
            duration_sec = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 1)
            status       = $Status
            report       = $ReportPath
        } | Export-Csv -Path $CsvPath -Append -NoTypeInformation -Encoding UTF8
    }
}

Write-Host "E4 finished: $($Total - $Failures)/$Total ok; ledger: $CsvPath"
if ($Failures -gt 0) { exit 1 }
