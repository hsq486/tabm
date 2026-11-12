# Extend the frozen protocol. Do not rerun seeds 0-2.
#
# Part A, five-seed main comparison:
#   TabM and MLP only, datasets california / churn / house, seeds 3 and 4.
#   Same configs as E2 (official MLP) and E3 (reduced TabM). Threads stay at 14.
#
# Part B, E8 label noise, pre-registered before looking at the new numbers:
#   Dataset california. Methods MLP and TabM. Seeds 0, 1, 2.
#   Training labels only. Regression noise is Gaussian with standard deviation
#   equal to 10% or 20% of the clean training-label standard deviation,
#   added after standardization on the clean training labels.
#   Validation and test labels stay clean. The 0% row is the existing E2/E3
#   run of the same config and seed; it is not trained again.
#   Support requires TabM to keep a lower mean RMSE than MLP at 10%, with at
#   least two of the three seeds pointing the same way. The 20% row is reported
#   either way and does not have to preserve the gap.

param(
    [string]$Python = "D:\Anaconda\envs\tabm\python.exe",
    # Generate the four E8 configs into $env:TEMP and exit without training.
    # Used to inspect the produced TOML before spending CPU time.
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Model = Join-Path $Root "bin\model.py"
$ConfigsDir = Join-Path $Root "configs"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

$Commit = (git -C $Root rev-parse --short HEAD).Trim()
$env:OMP_NUM_THREADS = "14"
$env:MKL_NUM_THREADS = "14"

function Invoke-Train {
    param(
        [string]$LedgerDir,
        [string]$RunDir,
        [string]$ConfigText,
        [string]$Method,
        [string]$Dataset,
        [string]$Seed,
        [string]$ExtraColumns
    )

    New-Item -ItemType Directory -Force $LedgerDir, $RunDir | Out-Null
    $CsvPath = Join-Path $LedgerDir "runs.csv"
    if (-not (Test-Path $CsvPath)) {
        $Header = "timestamp,commit,method,dataset,seed,exit_code,duration_sec,status,report"
        if ($ExtraColumns) {
            $Header = "timestamp,commit,method,dataset,noise,seed,exit_code,duration_sec,status,report"
        }
        [System.IO.File]::WriteAllText($CsvPath, "$Header`r`n", $Utf8NoBom)
    }

    $ConfigPath = Join-Path $RunDir "$Seed.toml"
    $LogPath = Join-Path $RunDir "$Seed.log"
    $DonePath = Join-Path $RunDir "$Seed\DONE"
    $ReportPath = Join-Path $RunDir "$Seed\report.json"
    [System.IO.File]::WriteAllText($ConfigPath, $ConfigText, $Utf8NoBom)

    if (Test-Path $DonePath) {
        Write-Host "skip $Method / $Dataset / seed=$Seed (already done)"
        return $true
    }

    Write-Host "run $Method / $Dataset / seed=$Seed"
    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $SavedErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python $Model $ConfigPath --force 2>&1 |
            ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ }
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

    $Status = if ($ExitCode -eq 0 -and (Test-Path $DonePath)) { "ok" } else { "failed" }
    $Relative = "paper/" + $ReportPath.Substring($Root.Length + 1).Replace("\", "/")
    $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Duration = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 1)
    if ($ExtraColumns) {
        $Line = "$Stamp,$Commit,$Method,$Dataset,$ExtraColumns,$Seed,$ExitCode,$Duration,$Status,$Relative"
    }
    else {
        $Line = "$Stamp,$Commit,$Method,$Dataset,$Seed,$ExitCode,$Duration,$Status,$Relative"
    }
    [System.IO.File]::AppendAllText($CsvPath, "$Line`r`n", $Utf8NoBom)
    Write-Host "$Status in $Duration s -> $Relative"
    return ($Status -eq "ok")
}

# Insert `label_noise` as a TOP-LEVEL key, immediately after the top-level
# `amp = true` line. Appending the key to the end of the file would place it
# inside the last `[...]` table (both source configs end with
# `[model.backbone]`), where `config.get('label_noise')` in bin/model.py --
# which reads the top level -- would never see it. In that case the run would
# either fail with an unexpected backbone kwarg or, worse, silently train on
# clean labels and report it as a noise run.
function Set-LabelNoise {
    param(
        [string]$Text,
        [string]$Level,
        [string]$Source
    )

    if ($Text -match '(?m)^\s*label_noise\s*=') {
        throw "label_noise is already present in $Source"
    }

    $Lines = $Text -split "\r?\n"
    $Anchor = -1
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i] -match '^amp\s*=\s*true\s*$') {
            $Anchor = $i
            break
        }
    }
    if ($Anchor -lt 0) {
        throw "Top-level 'amp = true' anchor not found in $Source; refusing to guess where label_noise belongs"
    }

    $Out = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        $Out.Add($Lines[$i])
        if ($i -eq $Anchor) {
            $Out.Add("label_noise = $Level")
        }
    }
    return ($Out -join "`r`n")
}

$Failures = 0

if ($PreflightOnly) {
    $OutDir = Join-Path $env:TEMP "e8-preflight"
    New-Item -ItemType Directory -Force $OutDir | Out-Null
    $Cases = @(
        @{ Name = "mlp"; Source = (Join-Path $Root "exp\mlp\california\0-evaluation\0.toml") },
        @{ Name = "tabm"; Source = (Join-Path $ConfigsDir "E3-tabm-california.toml") }
    )
    foreach ($Case in $Cases) {
        foreach ($Level in @("0.1", "0.2")) {
            $Text = [System.IO.File]::ReadAllText($Case.Source)
            $Text = Set-LabelNoise -Text $Text -Level $Level -Source $Case.Source
            $KeyPos = $Text.IndexOf("`nlabel_noise")
            $TablePos = $Text.IndexOf("`n[")
            if ($KeyPos -lt 0 -or $TablePos -lt 0 -or $KeyPos -gt $TablePos) {
                throw "preflight failed: label_noise is not a top-level key for $($Case.Name) at $Level"
            }
            $Path = Join-Path $OutDir "$($Case.Name)-noise=$Level.toml"
            [System.IO.File]::WriteAllText($Path, $Text, $Utf8NoBom)
            Write-Host "preflight ok -> $Path"
        }
    }
    exit 0
}

Write-Host "=== seeds 3 and 4: MLP and TabM ==="
foreach ($Dataset in @("california", "churn", "house")) {
    $MlpSource = Join-Path $Root "exp\mlp\$Dataset\0-evaluation\0.toml"
    $TabmSource = Join-Path $ConfigsDir "E3-tabm-$Dataset.toml"
    foreach ($Seed in @(3, 4)) {
        $MlpText = [System.IO.File]::ReadAllText($MlpSource)
        $MlpText = $MlpText -replace "(?m)^seed\s*=\s*\d+", "seed = $Seed"
        $Ok = Invoke-Train `
            -LedgerDir (Join-Path $Root "results\E2") `
            -RunDir (Join-Path $Root "exp\reproduce\E2-baseline\mlp\$Dataset") `
            -ConfigText $MlpText `
            -Method "mlp" -Dataset $Dataset -Seed "$Seed" -ExtraColumns ""
        if (-not $Ok) { $Failures += 1 }

        $TabmText = [System.IO.File]::ReadAllText($TabmSource)
        $TabmText = $TabmText -replace "(?m)^seed\s*=\s*\d+", "seed = $Seed"
        $Ok = Invoke-Train `
            -LedgerDir (Join-Path $Root "results\E3") `
            -RunDir (Join-Path $Root "exp\reproduce\E3\tabm\$Dataset") `
            -ConfigText $TabmText `
            -Method "tabm" -Dataset $Dataset -Seed "$Seed" -ExtraColumns ""
        if (-not $Ok) { $Failures += 1 }
    }
}

Write-Host "=== E8 label noise on california ==="
foreach ($Level in @("0.1", "0.2")) {
    foreach ($Seed in @(0, 1, 2)) {
        $MlpSource = Join-Path $Root "exp\mlp\california\0-evaluation\0.toml"
        $MlpText = [System.IO.File]::ReadAllText($MlpSource)
        $MlpText = $MlpText -replace "(?m)^seed\s*=\s*\d+", "seed = $Seed"
        $MlpText = Set-LabelNoise -Text $MlpText -Level $Level -Source $MlpSource
        $Ok = Invoke-Train `
            -LedgerDir (Join-Path $Root "results\E8") `
            -RunDir (Join-Path $Root "exp\reproduce\E8\mlp\california\noise=$Level") `
            -ConfigText $MlpText `
            -Method "mlp" -Dataset "california" -Seed "$Seed" -ExtraColumns $Level
        if (-not $Ok) { $Failures += 1 }

        $TabmSource = Join-Path $ConfigsDir "E3-tabm-california.toml"
        $TabmText = [System.IO.File]::ReadAllText($TabmSource)
        $TabmText = $TabmText -replace "(?m)^seed\s*=\s*\d+", "seed = $Seed"
        $TabmText = Set-LabelNoise -Text $TabmText -Level $Level -Source $TabmSource
        $Ok = Invoke-Train `
            -LedgerDir (Join-Path $Root "results\E8") `
            -RunDir (Join-Path $Root "exp\reproduce\E8\tabm\california\noise=$Level") `
            -ConfigText $TabmText `
            -Method "tabm" -Dataset "california" -Seed "$Seed" -ExtraColumns $Level
        if (-not $Ok) { $Failures += 1 }
    }
}

if ($Failures -gt 0) {
    Write-Host "EXTENSIONS FAILED: $Failures run(s)"
    exit 1
}
Write-Host "EXTENSIONS DONE"
