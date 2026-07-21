$ErrorActionPreference = "Stop"

$PythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$TrainingScript = Join-Path `
    $PSScriptRoot `
    "train_vasu_60m_fineweb_blocks.py"
$CheckpointDir = Join-Path `
    $PSScriptRoot `
    "checkpoints\vasu_60m\fineweb_blocks"

$MaximumRuntimeHours = 11
$NormalCooldownSeconds = 10
$ThermalCooldownSeconds = 600
$MinimumRemainingMinutes = 5
$TargetGlobalStep = 152000

if (-not (Test-Path $PythonPath)) {
    throw "Python executable not found: $PythonPath"
}

if (-not (Test-Path $TrainingScript)) {
    throw "Training script not found: $TrainingScript"
}

if (-not (Test-Path $CheckpointDir)) {
    throw "Checkpoint directory not found: $CheckpointDir"
}


function Get-LatestValidCheckpoint {
    $InspectorPath = Join-Path $env:TEMP "vasu_checkpoint_inspector.py"

    $InspectorCode = @"
from pathlib import Path
import sys
import torch

checkpoint_dir = Path(sys.argv[1])

required_keys = {
    "epoch",
    "global_step",
    "model",
    "optimizer",
    "scheduler",
    "loss",
}

candidates = sorted(
    checkpoint_dir.glob("*.pt"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

valid_checkpoints = []

for path in candidates:
    if path.name.endswith(".tmp"):
        continue

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not required_keys.issubset(checkpoint):
            continue

        valid_checkpoints.append(
            (
                int(checkpoint["global_step"]),
                str(path),
            )
        )

    except Exception:
        continue

if not valid_checkpoints:
    raise SystemExit(1)

global_step, checkpoint_path = max(
    valid_checkpoints,
    key=lambda item: item[0],
)

print(f"{global_step}|{checkpoint_path}")
"@

    Set-Content `
        -Path $InspectorPath `
        -Value $InspectorCode `
        -Encoding UTF8

    try {
        $inspectionOutput = & $PythonPath `
            $InspectorPath `
            $CheckpointDir

        if (
            $LASTEXITCODE -ne 0 -or
            [string]::IsNullOrWhiteSpace($inspectionOutput)
        ) {
            throw "Failed to inspect latest valid checkpoint."
        }

        $outputLine = @($inspectionOutput)[-1].Trim()
        $parts = $outputLine.Split("|", 2)

        if ($parts.Count -ne 2) {
            throw "Unexpected checkpoint output: $outputLine"
        }

        return [PSCustomObject]@{
            GlobalStep = [int]$parts[0]
            Path = $parts[1]
            IsThermal = (
                [System.IO.Path]::GetFileName($parts[1]) -like
                "thermal_stop_step_*.pt"
            )
        }
    }
    finally {
        Remove-Item `
            -Path $InspectorPath `
            -Force `
            -ErrorAction SilentlyContinue
    }
}


function Get-FreeDiskGiB {
    $driveRoot = [System.IO.Path]::GetPathRoot($CheckpointDir)
    $drive = [System.IO.DriveInfo]::new($driveRoot)

    return $drive.AvailableFreeSpace / 1GB
}


$StartTime = Get-Date
$EndDeadline = $StartTime.AddHours($MaximumRuntimeHours)
$InitialCheckpoint = Get-LatestValidCheckpoint
$StartingGlobalStep = $InitialCheckpoint.GlobalStep

$FinalCheckpoint = $InitialCheckpoint
$EndReason = "time limit"
$UserInterrupted = $false

Write-Host ""
Write-Host "VASU-60M 11-hour training runner" -ForegroundColor Cyan
Write-Host "Start time: $StartTime"
Write-Host "Deadline: $EndDeadline"
Write-Host "Target global step: $TargetGlobalStep"
Write-Host "Starting global step: $StartingGlobalStep"
Write-Host "Starting checkpoint: $($InitialCheckpoint.Path)"
Write-Host ""

try {
    while ($true) {
        $Now = Get-Date
        $Elapsed = $Now - $StartTime
        $Remaining = $EndDeadline - $Now

        if ($Now -ge $EndDeadline) {
            $EndReason = "time limit"
            break
        }

        if ($Remaining.TotalMinutes -lt $MinimumRemainingMinutes) {
            $EndReason = "less than five minutes remaining"
            break
        }

        $BeforeCheckpoint = Get-LatestValidCheckpoint
        $CurrentGlobalStep = [int64]$BeforeCheckpoint.GlobalStep
        $StepsRemainingToTarget = [Math]::Max(
            $TargetGlobalStep - $CurrentGlobalStep,
            0
        )

        Write-Host "Target global step: $TargetGlobalStep"
        Write-Host "Current global step: $CurrentGlobalStep"
        Write-Host "Remaining optimizer steps: $StepsRemainingToTarget"

        if ($CurrentGlobalStep -ge $TargetGlobalStep) {
            $FinalCheckpoint = $BeforeCheckpoint
            $EndReason = "target global step reached"
            Write-Host `
                "Target global step reached before starting a block." `
                -ForegroundColor Green
            break
        }

        $FreeDiskGiB = Get-FreeDiskGiB

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "Current time: $Now"
        Write-Host "Elapsed: $Elapsed"
        Write-Host "Remaining: $Remaining"
        Write-Host "Latest checkpoint: $($BeforeCheckpoint.Path)"
        Write-Host "Latest global step: $($BeforeCheckpoint.GlobalStep)"
        Write-Host ("Free disk: {0:N2} GiB" -f $FreeDiskGiB)
        Write-Host "Starting one 200-step training block..." `
            -ForegroundColor Cyan

        & $PythonPath $TrainingScript
        $PythonExitCode = $LASTEXITCODE

        if ($PythonExitCode -ne 0) {
            $EndReason = "training failure"
            Write-Host `
                "Training exited with code $PythonExitCode." `
                -ForegroundColor Red
            break
        }

        $AfterCheckpoint = Get-LatestValidCheckpoint
        $FinalCheckpoint = $AfterCheckpoint

        Write-Host ""
        Write-Host "Python exit code: $PythonExitCode"
        Write-Host "Latest checkpoint: $($AfterCheckpoint.Path)"
        Write-Host "Latest global step: $($AfterCheckpoint.GlobalStep)"

        $ProgressMade = (
            $AfterCheckpoint.GlobalStep -gt
            $BeforeCheckpoint.GlobalStep
        )

        Write-Host "Progress advanced: $ProgressMade"

        if (-not $ProgressMade) {
            $EndReason = "lack of progress"
            Write-Host `
                "Global step did not increase. Stopping." `
                -ForegroundColor Red
            break
        }

        $StepsRemainingToTarget = [Math]::Max(
            $TargetGlobalStep - $AfterCheckpoint.GlobalStep,
            0
        )
        Write-Host "Target global step: $TargetGlobalStep"
        Write-Host "Current global step: $($AfterCheckpoint.GlobalStep)"
        Write-Host "Remaining optimizer steps: $StepsRemainingToTarget"

        if ($AfterCheckpoint.GlobalStep -ge $TargetGlobalStep) {
            $EndReason = "target global step reached"
            Write-Host `
                "Target global step reached after the completed block." `
                -ForegroundColor Green
            break
        }

        $Now = Get-Date

        if ($Now -ge $EndDeadline) {
            $EndReason = "time limit"
            break
        }

        if ($AfterCheckpoint.IsThermal) {
            $CooldownSeconds = $ThermalCooldownSeconds
            Write-Host `
                "Thermal-stop checkpoint detected." `
                -ForegroundColor Yellow
            Write-Host `
                "Cooling for 10 minutes before resuming..." `
                -ForegroundColor Yellow
        }
        else {
            $CooldownSeconds = $NormalCooldownSeconds
            Write-Host `
                "Cooling for 10 minutes before the next block..." `
                -ForegroundColor Yellow
        }

        $RemainingSeconds = (
            $EndDeadline - (Get-Date)
        ).TotalSeconds

        if ($RemainingSeconds -le 0) {
            $EndReason = "time limit"
            break
        }

        $ActualCooldownSeconds = [Math]::Min(
            $CooldownSeconds,
            [int]$RemainingSeconds
        )

        Start-Sleep -Seconds $ActualCooldownSeconds
    }
}
catch [System.Management.Automation.PipelineStoppedException] {
    $UserInterrupted = $true
    $EndReason = "user interruption"
}
catch {
    $EndReason = "training failure"
    Write-Host $_ -ForegroundColor Red
}
finally {
    $FinishTime = Get-Date
    $TotalElapsed = $FinishTime - $StartTime

    try {
        $FinalCheckpoint = Get-LatestValidCheckpoint
    }
    catch {
        # Keep the last checkpoint found before the error.
    }

    $FinalGlobalStep = $FinalCheckpoint.GlobalStep
    $CompletedSteps = $FinalGlobalStep - $StartingGlobalStep
    $TargetReached = $FinalGlobalStep -ge $TargetGlobalStep
    $OvershootAmount = [Math]::Max(
        $FinalGlobalStep - $TargetGlobalStep,
        0
    )

    Write-Host ""
    Write-Host "VASU-60M 11-HOUR RUN SUMMARY" -ForegroundColor Cyan
    Write-Host "Start time: $StartTime"
    Write-Host "End time: $FinishTime"
    Write-Host "Total elapsed time: $TotalElapsed"
    Write-Host "Target global step: $TargetGlobalStep"
    Write-Host "Starting global step: $StartingGlobalStep"
    Write-Host "Final global step: $FinalGlobalStep"
    Write-Host "Total optimizer steps completed: $CompletedSteps"
    Write-Host "Target reached: $TargetReached"
    Write-Host "Overshoot amount: $OvershootAmount"
    Write-Host "Latest valid checkpoint: $($FinalCheckpoint.Path)"
    Write-Host "End reason: $EndReason"
}

