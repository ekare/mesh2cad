param(
    [string]$Python = "python",
    [string]$WorkRoot = "",
    [string]$ModelCheckpoint = "",
    [string]$ModelCard = ""
)

$ErrorActionPreference = "Stop"

function Assert-ExternalSuccess([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $WorkRoot) { $WorkRoot = Join-Path $repo ".build\windows-x64" }
$work = [IO.Path]::GetFullPath($WorkRoot)
$release = Join-Path $repo "release"
$venv = Join-Path $work "venv"
$dist = Join-Path $work "dist"
$pyiWork = Join-Path $work "pyinstaller"
$neuralRelease = -not [string]::IsNullOrWhiteSpace($ModelCheckpoint)
$packageName = if ($neuralRelease) {
    "mesh2cad-0.1.0-windows-x64-neural"
} else {
    "mesh2cad-0.1.0-windows-x64"
}
$package = Join-Path $dist $packageName

$volumeRoot = [IO.Path]::GetPathRoot($work).TrimEnd('\')
if ($work.TrimEnd('\') -eq $volumeRoot -or $work -eq $repo) {
    throw "Unsafe release work path: $work"
}
foreach ($target in @($venv, $dist, $pyiWork)) {
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
}
New-Item -ItemType Directory -Force -Path $work, $release | Out-Null
foreach ($asset in @(
    "index.html", "app.css", "app.js", "favicon.svg",
    "dashboard.html", "dashboard.css", "dashboard.js"
)) {
    $assetPath = Join-Path $repo "src\mesh2cad\web_static\$asset"
    if (-not (Test-Path -LiteralPath $assetPath)) {
        throw "Missing embedded web asset: $assetPath"
    }
}

& $Python -m venv $venv
Assert-ExternalSuccess "Release virtual environment creation"
$venvPython = Join-Path $venv "Scripts\python.exe"
& $venvPython -c "import struct; assert struct.calcsize('P') * 8 == 64, 'A 64-bit Python is required'"
Assert-ExternalSuccess "64-bit Python check"
& $venvPython -m pip install --upgrade pip
Assert-ExternalSuccess "pip upgrade"
& $venvPython -m pip install -e "${repo}[web,dev,release]"
Assert-ExternalSuccess "Release dependency installation"
if ($neuralRelease) {
    $resolvedModel = (Resolve-Path -LiteralPath $ModelCheckpoint).Path
    if ((Split-Path -Leaf $resolvedModel) -ne "mesh2cad-automate-v0.1.pt") {
        throw "The bundled public checkpoint must be named mesh2cad-automate-v0.1.pt"
    }
    if ([string]::IsNullOrWhiteSpace($ModelCard)) {
        throw "A neural release requires -ModelCard"
    }
    $resolvedModelCard = (Resolve-Path -LiteralPath $ModelCard).Path
    $modelArtifacts = Split-Path -Parent $resolvedModelCard
    $requiredModelArtifacts = @(
        "automate-v0.1-16k-split.json",
        "automate-v0.1-16k-provenance.json",
        "mesh2cad-automate-v0.1-benchmark.json",
        "mesh2cad-automate-v0.1-repair-benchmark.json",
        "mesh2cad-automate-v0.1-qualification.json",
        "SHA256SUMS"
    )
    foreach ($artifactName in $requiredModelArtifacts) {
        if (-not (Test-Path -LiteralPath (Join-Path $modelArtifacts $artifactName))) {
            throw "Missing required neural release artifact: $artifactName"
        }
    }
    $qualification = Get-Content -Raw -LiteralPath (Join-Path $modelArtifacts "mesh2cad-automate-v0.1-qualification.json") | ConvertFrom-Json
    $modelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedModel).Hash.ToLowerInvariant()
    if (-not $qualification.passed -or $qualification.checkpoint_sha256 -ne $modelHash) {
        throw "Neural release qualification is missing, failed, or belongs to another checkpoint"
    }
    & $venvPython -m pip install "torch>=2.3,<3" --index-url https://download.pytorch.org/whl/cpu
    Assert-ExternalSuccess "CPU PyTorch installation"
    & $venvPython -c "import sys, torch; p=sys.argv[1]; c=torch.load(p,map_location='cpu',weights_only=False); assert c.get('checkpoint_kind') == 'mesh2cad-inference'; assert c.get('training_summary',{}).get('training_complete') is True" $resolvedModel
    Assert-ExternalSuccess "Public inference checkpoint validation"
}
& $venvPython -m pip check
Assert-ExternalSuccess "Release dependency check"

Push-Location $repo
try {
    & $venvPython -m ruff check .
    Assert-ExternalSuccess "Python lint"
    & $venvPython -m pytest -q
    Assert-ExternalSuccess "Python tests"
    if ($neuralRelease) { $env:MESH2CAD_BUNDLE_CHECKPOINT = $resolvedModel }
    & $venvPython -m PyInstaller --noconfirm --clean --distpath $dist --workpath $pyiWork mesh2cad-windows.spec
    Assert-ExternalSuccess "PyInstaller build"
} finally {
    Remove-Item Env:MESH2CAD_BUNDLE_CHECKPOINT -ErrorAction SilentlyContinue
    Pop-Location
}

Copy-Item -LiteralPath (Join-Path $repo "LICENSE") -Destination $package
Copy-Item -LiteralPath (Join-Path $repo "DISCLAIMER.md") -Destination $package
Copy-Item -LiteralPath (Join-Path $repo "THIRD_PARTY_NOTICES.md") -Destination $package
Copy-Item -LiteralPath (Join-Path $repo "README.md") -Destination $package
if ($neuralRelease) {
    Copy-Item -LiteralPath $resolvedModelCard -Destination (Join-Path $package "MODEL_CARD.md")
    $modelNoticeDirectory = Join-Path $package "model"
    New-Item -ItemType Directory -Force -Path $modelNoticeDirectory | Out-Null
    foreach ($artifactName in $requiredModelArtifacts) {
        Copy-Item -LiteralPath (Join-Path $modelArtifacts $artifactName) -Destination $modelNoticeDirectory
    }
}
& $venvPython (Join-Path $repo "scripts\collect_licenses.py") (Join-Path $package "licenses") --repository $repo
Assert-ExternalSuccess "Third-party license collection"

$port = 18765
$server = Start-Process -FilePath (Join-Path $package "mesh2cad-web.exe") -ArgumentList "--no-browser", "--port", $port, "--strict-port" -PassThru -WindowStyle Hidden
try {
    $smokeArguments = @(
        (Join-Path $repo "scripts\smoke_release.py"),
        "http://127.0.0.1:$port",
        (Join-Path $repo "examples\cube_with_hole\cube_with_hole.stl"),
        (Join-Path $work "smoke.step")
    )
    if ($neuralRelease) {
        $smokeArguments += @("--expect-engine", "neural", "--expect-checkpoint", "loaded")
    } else {
        $smokeArguments += @("--expect-engine", "auto", "--expect-checkpoint", "none")
    }
    & $venvPython @smokeArguments
    Assert-ExternalSuccess "Packaged web conversion smoke test"
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
& (Join-Path $package "mesh2cad.exe") (Join-Path $repo "examples\cube_with_hole\cube_with_hole.stl") -o (Join-Path $work "cli-smoke.step")
Assert-ExternalSuccess "Packaged CLI conversion smoke test"
& (Join-Path $package "mesh2cad-repair.exe") (Join-Path $repo "examples\cube_with_hole\cube_with_hole.stl") -o (Join-Path $work "repair-smoke.stl")
Assert-ExternalSuccess "Packaged repair CLI smoke test"
if (-not (Test-Path -LiteralPath (Join-Path $work "repair-smoke.repair.json"))) {
    throw "Packaged repair CLI did not write its JSON report"
}
& $venvPython (Join-Path $repo "scripts\check_personal_data.py") $package
Assert-ExternalSuccess "Packaged personal-data scan"

$archive = Join-Path $release "$packageName.zip"
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
Compress-Archive -LiteralPath $package -DestinationPath $archive -CompressionLevel Optimal
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
Set-Content -LiteralPath "$archive.sha256" -Value "$hash  $(Split-Path $archive -Leaf)" -Encoding ascii
Write-Host "Release: $archive"
Write-Host "SHA-256: $hash"
