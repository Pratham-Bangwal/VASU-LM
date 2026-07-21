
$ErrorActionPreference = "Stop"

$PythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$TrainingScript = Join-Path `
    $PSScriptRoot `
    "train_vasu_60m_fineweb_blocks.py"
$CheckpointDir = Join-Path `
    $PSScriptRoot `
    "checkpoints\vasu_60m\fineweb_blocks"

$TargetGlobalStep = 10000
$WaitSeconds = 10

if (-not (Test-Path $PythonPath)) {
    throw "Python executable not found: $PythonPath"
}

if (-not (Test-Path $TrainingScript)) {
    throw "Training script not found: $TrainingScript"
}

if (-not (Test-Path $CheckpointDir)) {
    throw "Checkpoint directory not found: $CheckpointDir"
}

function Get-LatestCheckpointInfo {
    $checkpoint = Get-ChildItem `
        -Path $CheckpointDir `
        -Filter "*.pt" `
        -File |
        ForEach-Object {
            if ($_.BaseName -match "(\d+)$") {
                [PSCustomObject]@{
                    GlobalStep = [int]$Matches[1]
                    Path = $_.FullName
                }
            }
        } |
        Sort-Object GlobalStep -Descending |
        Select-Object -First 1

    if ($null -eq $checkpoint) {
        throw "No checkpoint files found in: $CheckpointDir"
    }

    return $checkpoint
}

while ($true) {
    $checkpoint = Get-LatestCheckpointInfo

    Write-Host ""
    Write-Host "Latest checkpoint: $($checkpoint.Path)"
    Write-Host "Latest global step: $($checkpoint.GlobalStep)"

    if ($checkpoint.GlobalStep -ge $TargetGlobalStep) {
        Write-Host `
            "Target global step $TargetGlobalStep reached." `
            -ForegroundColor Green
        break
    }

    Write-Host `
        "Starting one 100-step training block..." `
        -ForegroundColor Cyan

    & $PythonPath $TrainingScript

    if ($LASTEXITCODE -ne 0) {
        Write-Host `
            "Training failed with exit code $LASTEXITCODE." `
            -ForegroundColor Red
        break
    }

    $checkpoint = Get-LatestCheckpointInfo

    Write-Host ""
    Write-Host "Completed global step: $($checkpoint.GlobalStep)"
    Write-Host "Latest checkpoint: $($checkpoint.Path)"

    if ($checkpoint.GlobalStep -ge $TargetGlobalStep) {
        Write-Host `
            "Target global step $TargetGlobalStep reached." `
            -ForegroundColor Green
        break
    }

    Write-Host `
        "Cooling for $WaitSeconds seconds..." `
        -ForegroundColor Yellow

    Start-Sleep -Seconds $WaitSeconds
}

