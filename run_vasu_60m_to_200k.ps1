param(
    [int64]$TargetGlobalStep = 200000,
    [int]$CooldownSeconds = 10,
    [int]$ThermalCooldownSeconds = 600,
    [double]$MinimumFreeDiskGiB = 10,
    [string]$PythonExecutable = ".\.venv\Scripts\python.exe",
    [string]$RunnerPath = ".\train_vasu_60m_fineweb_blocks.py",
    [string]$LogSubdirectory = "logs\vasu_60m_200k",
    [switch]$DryRun,
    [switch]$AutoResumeAfterThermalStop,
    [int]$ThermalRestartThresholdC = 70,
    [int]$MaximumThermalRestarts = 1
)

$ErrorActionPreference = "Stop"
$BlockSteps = 200
$ProjectRoot = $PSScriptRoot
$CheckpointDirectory = Join-Path `
    $ProjectRoot `
    "checkpoints\vasu_60m\fineweb_blocks"
$MilestoneDirectory = Join-Path `
    $ProjectRoot `
    "checkpoints\vasu_60m\milestones"
$InspectorPath = Join-Path `
    $ProjectRoot `
    "scripts\inspect_vasu_checkpoint.py"
$LogDirectory = Join-Path $ProjectRoot $LogSubdirectory
$SummaryPath = Join-Path $LogDirectory "run_summary.jsonl"

$PythonExecutable = [IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot $PythonExecutable)
)
$RunnerPath = [IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot $RunnerPath)
)


function Invoke-CheckpointInspector {
    param(
        [string[]]$Arguments
    )

    $errorPath = Join-Path `
        $env:TEMP `
        ("vasu_inspector_{0}.err" -f [guid]::NewGuid().ToString("N"))
    try {
        $output = & $PythonExecutable $InspectorPath @Arguments `
            2> $errorPath
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            $errorText = Get-Content $errorPath -Raw -ErrorAction SilentlyContinue
            throw "Checkpoint inspection failed: $errorText"
        }
        $jsonText = ($output | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($jsonText)) {
            throw "Checkpoint inspector returned no JSON."
        }
        return $jsonText | ConvertFrom-Json
    }
    finally {
        Remove-Item $errorPath -Force -ErrorAction SilentlyContinue
    }
}


function Get-LatestValidCheckpoint {
    return Invoke-CheckpointInspector -Arguments @(
        "--checkpoint-dir",
        $CheckpointDirectory
    )
}


function Test-OneCheckpoint {
    param(
        [string]$Path,
        [switch]$MetadataOnly
    )

    $arguments = @("--checkpoint", $Path)
    if ($MetadataOnly) {
        $arguments += @("--no-finite-check", "--no-hash")
    }
    return Invoke-CheckpointInspector -Arguments $arguments
}


function Get-FreeDiskGiB {
    $driveRoot = [IO.Path]::GetPathRoot($CheckpointDirectory)
    $drive = [IO.DriveInfo]::new($driveRoot)
    return $drive.AvailableFreeSpace / 1GB
}


function Get-GpuTemperature {
    try {
        $value = & nvidia-smi `
            --query-gpu=temperature.gpu `
            --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $value) {
            return [int](($value | Select-Object -First 1).Trim())
        }
    }
    catch {
        return $null
    }
    return $null
}


function Get-ConflictingTrainingProcesses {
    $trainingPatterns = @(
        '(?i)(python(?:\.exe)?|pythonw(?:\.exe)?).*?(train_vasu[^\s"'']*\.py|train_fineweb\.py|train_alpaca[^\s"'']*\.py|train_ultrachat[^\s"'']*\.py)',
        '(?i)powershell(?:\.exe)?.*?run_vasu_60m_(to_200k|for_11_hours|until_10000)\.ps1'
    )
    # The invoking shell can contain this runner's filename in its command line.
    # Exclude this process and its ancestors so that only independent processes
    # are treated as training conflicts.
    $excludedPids = [System.Collections.Generic.HashSet[int]]::new()
    $processById = @{}
    foreach ($process in @(Get-CimInstance Win32_Process)) {
        $processById[[int]$process.ProcessId] = $process
    }

    $ancestorPid = [int]$PID
    while ($ancestorPid -gt 0 -and $excludedPids.Add($ancestorPid)) {
        if (-not $processById.ContainsKey($ancestorPid)) {
            break
        }
        $ancestorPid = [int]$processById[$ancestorPid].ParentProcessId
    }

    return @(
        $processById.Values | Where-Object {
            if ($excludedPids.Contains([int]$_.ProcessId) -or -not $_.CommandLine) {
                return $false
            }
            foreach ($pattern in $trainingPatterns) {
                if ($_.CommandLine -match $pattern) {
                    return $true
                }
            }
            return $false
        }
    )
}


function Assert-NoTrainingConflict {
    $conflicts = @(Get-ConflictingTrainingProcesses)
    if ($conflicts.Count -gt 0) {
        Write-Host "Conflicting training process detected:" -ForegroundColor Red
        foreach ($process in $conflicts) {
            Write-Host "  PID $($process.ProcessId): $($process.CommandLine)"
        }
        throw "Refusing to launch while another training process is active."
    }
}


function Get-LastMatch {
    param(
        [string]$Text,
        [string]$Pattern,
        [ValidateSet("int", "double", "bool", "string")]
        [string]$Type = "string"
    )

    $matches = [regex]::Matches($Text, $Pattern)
    if ($matches.Count -eq 0) {
        return $null
    }
    $value = $matches[$matches.Count - 1].Groups[1].Value
    switch ($Type) {
        "int" { return [int64]$value }
        "double" { return [double]::Parse(
            $value,
            [Globalization.CultureInfo]::InvariantCulture
        ) }
        "bool" { return $value -eq "True" }
        default { return $value }
    }
}


function Write-SummaryRecord {
    param(
        [hashtable]$Record
    )

    $json = $Record | ConvertTo-Json -Compress -Depth 6
    Add-Content -Path $SummaryPath -Value $json -Encoding UTF8
}


function Assert-RequiredMilestones {
    $required = @(150000, 152000, 152200)
    foreach ($step in $required) {
        $path = Join-Path `
            $MilestoneDirectory `
            ("fineweb_step_{0}.pt" -f $step)
        if (-not (Test-Path $path -PathType Leaf)) {
            if ($step -eq 152200) {
                Write-Host "Required step-152200 milestone is missing." `
                    -ForegroundColor Red
                Write-Host "Preserve it before running automation with:"
                Write-Host (
                    "Copy-Item .\checkpoints\vasu_60m\fineweb_blocks\" +
                    "block_final_step_152200.pt " +
                    ".\checkpoints\vasu_60m\milestones\" +
                    "fineweb_step_152200.pt"
                )
            }
            throw "Required milestone not found: $path"
        }
        $inspection = Test-OneCheckpoint -Path $path -MetadataOnly
        if (-not $inspection.valid -or $inspection.global_step -ne $step) {
            throw "Required milestone is invalid or has the wrong step: $path"
        }
    }
}


function Get-PythonHardCeiling {
    $runnerText = Get-Content $RunnerPath -Raw
    $match = [regex]::Match(
        $runnerText,
        "TARGET_GLOBAL_STEP\s*=\s*([0-9_]+)"
    )
    if (-not $match.Success) {
        throw "Could not determine Python runner hard ceiling."
    }
    return [int64]($match.Groups[1].Value.Replace("_", ""))
}


function Preserve-FinalMilestone {
    param(
        [object]$Checkpoint
    )

    $destination = Join-Path `
        $MilestoneDirectory `
        ("fineweb_step_{0}.pt" -f $TargetGlobalStep)
    if (Test-Path $destination) {
        $existing = Test-OneCheckpoint -Path $destination
        if (
            -not $existing.valid -or
            $existing.global_step -ne $TargetGlobalStep -or
            $existing.sha256 -ne $Checkpoint.sha256
        ) {
            throw "Existing final milestone does not match the final checkpoint."
        }
        return $existing
    }
    return Invoke-CheckpointInspector -Arguments @(
        "--checkpoint",
        $Checkpoint.path,
        "--preserve-milestone",
        $destination,
        "--expected-step",
        "$TargetGlobalStep"
    )
}


if (-not (Test-Path $PythonExecutable -PathType Leaf)) {
    throw "Python executable not found: $PythonExecutable"
}
if (-not (Test-Path $RunnerPath -PathType Leaf)) {
    throw "Training runner not found: $RunnerPath"
}
if (-not (Test-Path $InspectorPath -PathType Leaf)) {
    throw "Checkpoint inspector not found: $InspectorPath"
}
if (-not (Test-Path $CheckpointDirectory -PathType Container)) {
    throw "Checkpoint directory not found: $CheckpointDirectory"
}

$PythonHardCeiling = Get-PythonHardCeiling
if ($PythonHardCeiling -lt $TargetGlobalStep) {
    throw (
        "Python hard ceiling $PythonHardCeiling is below wrapper target " +
        "$TargetGlobalStep."
    )
}

New-Item $LogDirectory -ItemType Directory -Force | Out-Null
Assert-RequiredMilestones
Assert-NoTrainingConflict
$InitialCheckpoint = Get-LatestValidCheckpoint
$StartingGlobalStep = [int64]$InitialCheckpoint.global_step
if ($StartingGlobalStep -gt $TargetGlobalStep) {
    throw "Latest global step is above target: $StartingGlobalStep"
}

$initialRemaining = $TargetGlobalStep - $StartingGlobalStep
if (
    $TargetGlobalStep -lt $PythonHardCeiling -and
    $initialRemaining -gt 0 -and
    $initialRemaining % $BlockSteps -ne 0
) {
    throw (
        "A target below the Python ceiling must align to the fixed " +
        "$BlockSteps-step block size."
    )
}

$InitialFreeDisk = Get-FreeDiskGiB
if ($InitialFreeDisk -lt $MinimumFreeDiskGiB) {
    throw (
        "Free disk $($InitialFreeDisk.ToString('N2')) GiB is below " +
        "$MinimumFreeDiskGiB GiB."
    )
}

Write-Host "VASU-60M bounded FineWeb orchestration" -ForegroundColor Cyan
Write-Host "Latest valid checkpoint: $($InitialCheckpoint.path)"
Write-Host "Current global step: $StartingGlobalStep"
Write-Host "Target global step: $TargetGlobalStep"
Write-Host "Python hard ceiling: $PythonHardCeiling"
Write-Host "Block size: $BlockSteps optimizer steps"
Write-Host ("Free disk: {0:N2} GiB" -f $InitialFreeDisk)
Write-Host "Summary log: $SummaryPath"

if ($DryRun) {
    $expectedNext = [Math]::Min(
        $StartingGlobalStep + $BlockSteps,
        $TargetGlobalStep
    )
    Write-Host "DRY RUN PASSED" -ForegroundColor Green
    Write-Host "Expected next step: $expectedNext"
    Write-Host "Subprocess command: $PythonExecutable $RunnerPath"
    Write-Host "No training process was launched."
    exit 0
}

$CurrentCheckpoint = $InitialCheckpoint
$BlocksCompleted = 0
$ThermalStops = 0
$ThermalRestarts = 0
$EndReason = "target not reached"
$Interrupted = $false

try {
    while ([int64]$CurrentCheckpoint.global_step -lt $TargetGlobalStep) {
        Assert-NoTrainingConflict
        $BeforeCheckpoint = Get-LatestValidCheckpoint
        $BeforeStep = [int64]$BeforeCheckpoint.global_step
        if ($BeforeStep -ge $TargetGlobalStep) {
            $CurrentCheckpoint = $BeforeCheckpoint
            $EndReason = "target global step reached"
            break
        }

        $FreeDiskBefore = Get-FreeDiskGiB
        if ($FreeDiskBefore -lt $MinimumFreeDiskGiB) {
            throw "Disk guard triggered before block: $FreeDiskBefore GiB"
        }
        $ExpectedEnd = [Math]::Min(
            $BeforeStep + $BlockSteps,
            $TargetGlobalStep
        )
        $timestamp = (Get-Date).ToUniversalTime().ToString(
            "yyyyMMddTHHmmssZ"
        )
        $logName = "block_{0}_to_{1}_{2}.log" -f `
            $BeforeStep, $ExpectedEnd, $timestamp
        $logPath = Join-Path $LogDirectory $logName
        $startedAt = (Get-Date).ToUniversalTime()

        Write-Host ""
        Write-Host "Starting validated block $BeforeStep -> $ExpectedEnd" `
            -ForegroundColor Cyan
        Write-Host "Checkpoint: $($BeforeCheckpoint.path)"
        Write-Host "SHA-256: $($BeforeCheckpoint.sha256)"
        Write-Host ("Free disk: {0:N2} GiB" -f $FreeDiskBefore)
        Write-Host "Log: $logPath"

        # Windows PowerShell 5 wraps native stderr (including tqdm progress)
        # as ErrorRecord objects. Keep it captured without promoting normal
        # progress output to a terminating PowerShell error.
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $PythonExecutable $RunnerPath 2>&1 |
                Tee-Object -FilePath $logPath |
                Out-Null
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorAction
        }
        $completedAt = (Get-Date).ToUniversalTime()
        $elapsedSeconds = ($completedAt - $startedAt).TotalSeconds
        $outputText = Get-Content $logPath -Raw

        if ($exitCode -ne 0) {
            $EndReason = "training failure"
            throw "Python block exited with status $exitCode. See $logPath"
        }

        $AfterCheckpoint = Get-LatestValidCheckpoint
        $AfterStep = [int64]$AfterCheckpoint.global_step
        if (
            $AfterCheckpoint.path -eq $BeforeCheckpoint.path -and
            $AfterStep -ne $BeforeStep
        ) {
            throw "Checkpoint path was unchanged while global step changed."
        }
        $delta = $AfterStep - $BeforeStep
        if ($delta -le 0) {
            throw "Global step did not increase."
        }
        if ($delta -gt $BlockSteps) {
            throw "Global step increased by more than $BlockSteps."
        }
        if ($AfterStep -gt $TargetGlobalStep) {
            throw "Training exceeded target global step."
        }
        $expectedDelta = $ExpectedEnd - $BeforeStep
        $thermalOutput = Get-LastMatch `
            -Text $outputText `
            -Pattern "Thermal stop occurred:\s*(True|False)" `
            -Type bool
        $thermalCheckpointPath = Join-Path `
            $CheckpointDirectory `
            ("thermal_stop_step_{0}.pt" -f $AfterStep)
        $ThermalStop = ($thermalOutput -eq $true) -or (
            Test-Path $thermalCheckpointPath -PathType Leaf
        )
        if (-not $ThermalStop -and $delta -ne $expectedDelta) {
            throw "Unexpected progress delta $delta; expected $expectedDelta."
        }

        if ($ThermalStop) {
            $thermalInspection = Test-OneCheckpoint -Path $thermalCheckpointPath
            if (
                -not $thermalInspection.valid -or
                $thermalInspection.global_step -ne $AfterStep
            ) {
                throw "Thermal checkpoint failed validation."
            }
            $AfterCheckpoint = $thermalInspection
            $ThermalStops += 1
        }

        $FreeDiskAfter = Get-FreeDiskGiB
        $diskDelta = $FreeDiskBefore - $FreeDiskAfter
        if ($FreeDiskAfter -lt $MinimumFreeDiskGiB) {
            throw "Disk guard triggered after block: $FreeDiskAfter GiB"
        }
        if ($diskDelta -gt 5) {
            Write-Host (
                "Warning: free disk fell by {0:N2} GiB in one block." -f
                $diskDelta
            ) -ForegroundColor Yellow
        }

        $metrics = @{
            started_at = $startedAt.ToString("o")
            completed_at = $completedAt.ToString("o")
            starting_step = $BeforeStep
            final_step = $AfterStep
            delta_steps = $delta
            checkpoint_path = $AfterCheckpoint.path
            checkpoint_sha256 = $AfterCheckpoint.sha256
            exit_code = $exitCode
            thermal_stop = $ThermalStop
            block_completed = Get-LastMatch `
                -Text $outputText `
                -Pattern "Block completed:\s*(True|False)" `
                -Type bool
            elapsed_seconds = [Math]::Round($elapsedSeconds, 3)
            maximum_temperature = Get-LastMatch `
                -Text $outputText `
                -Pattern "Maximum GPU temperature:\s*(\d+)\s*C" `
                -Type int
            peak_cuda_memory_mib = Get-LastMatch `
                -Text $outputText `
                -Pattern "CUDA peak memory:\s*([0-9.eE+-]+)\s*MiB" `
                -Type double
            train_loss = Get-LastMatch `
                -Text $outputText `
                -Pattern "Latest loss:\s*([0-9.eE+-]+)" `
                -Type double
            validation_loss_if_any = Get-LastMatch `
                -Text $outputText `
                -Pattern "Validation loss:\s*([0-9.eE+-]+)" `
                -Type double
            free_disk_before_gib = [Math]::Round($FreeDiskBefore, 3)
            free_disk_after_gib = [Math]::Round($FreeDiskAfter, 3)
            log_path = $logPath
        }
        Write-SummaryRecord -Record $metrics
        $CurrentCheckpoint = $AfterCheckpoint
        $BlocksCompleted += 1

        Write-Host "Validated final step: $AfterStep" -ForegroundColor Green
        Write-Host "Delta steps: $delta"
        Write-Host "Checkpoint: $($AfterCheckpoint.path)"
        Write-Host "Thermal stop: $ThermalStop"

        if ($AfterStep -eq $TargetGlobalStep) {
            $EndReason = "target global step reached"
            break
        }

        if ($ThermalStop) {
            if (-not $AutoResumeAfterThermalStop) {
                $EndReason = "thermal stop"
                break
            }
            if ($ThermalRestarts -ge $MaximumThermalRestarts) {
                $EndReason = "maximum thermal restart count reached"
                break
            }
            $ThermalRestarts += 1
            Write-Host (
                "Cooling for $ThermalCooldownSeconds seconds after thermal stop."
            ) -ForegroundColor Yellow
            Start-Sleep -Seconds $ThermalCooldownSeconds
            $temperature = Get-GpuTemperature
            if (
                $null -eq $temperature -or
                $temperature -gt $ThermalRestartThresholdC
            ) {
                $EndReason = "temperature remained above restart threshold"
                break
            }
            Test-OneCheckpoint -Path $AfterCheckpoint.path | Out-Null
        }
        else {
            Start-Sleep -Seconds $CooldownSeconds
        }
    }
}
catch [Management.Automation.PipelineStoppedException] {
    $Interrupted = $true
    $EndReason = "user interruption"
}
finally {
    try {
        $CurrentCheckpoint = Get-LatestValidCheckpoint
    }
    catch {
        Write-Host "Unable to inspect a final checkpoint: $_" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "VASU-60M ORCHESTRATION SUMMARY" -ForegroundColor Cyan
    Write-Host "Starting global step: $StartingGlobalStep"
    Write-Host "Current/final global step: $($CurrentCheckpoint.global_step)"
    Write-Host "Target global step: $TargetGlobalStep"
    Write-Host "Blocks completed: $BlocksCompleted"
    Write-Host "Thermal stops: $ThermalStops"
    Write-Host "Latest checkpoint: $($CurrentCheckpoint.path)"
    Write-Host "End reason: $EndReason"
}

if ([int64]$CurrentCheckpoint.global_step -eq $TargetGlobalStep) {
    $finalInspection = Get-LatestValidCheckpoint
    $milestone = Preserve-FinalMilestone -Checkpoint $finalInspection
    Write-Host "Final checkpoint validated: $($finalInspection.path)"
    Write-Host "Final checkpoint SHA-256: $($finalInspection.sha256)"
    if ($milestone.milestone) {
        Write-Host "Milestone: $($milestone.milestone.path)"
        Write-Host "Milestone SHA-256: $($milestone.milestone.sha256)"
    }
    else {
        Write-Host "Milestone: $($milestone.path)"
        Write-Host "Milestone SHA-256: $($milestone.sha256)"
    }
    exit 0
}

if ($Interrupted) {
    exit 130
}
exit 1
