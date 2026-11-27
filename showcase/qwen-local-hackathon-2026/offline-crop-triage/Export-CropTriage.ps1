param([string]$ProjectDir = 'D:\Organized\Projects\Qwen Local Hackathon')

$ErrorActionPreference = 'Stop'
$evidence = Join-Path $ProjectDir 'crop-training-evidence.zip'
if (-not (Test-Path -LiteralPath $evidence -PathType Leaf)) { throw "Missing $evidence" }
if ((Get-FileHash -LiteralPath $evidence -Algorithm SHA256).Hash -ne '60a97105fba89ee5f25bd6abac111864a1315e019141926f60f43793d87d2585') {
    throw 'The ZIP checksum differs from the verified training bundle.'
}
$work = Join-Path $ProjectDir 'crop-export-work'
New-Item -ItemType Directory -Path $work -Force | Out-Null

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
function Get-Download {
    param([string]$Url, [string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        Invoke-Checked 'curl.exe' @('--fail', '--location', '--retry', '3', '--output', "$Path.partial", $Url)
        Move-Item -LiteralPath "$Path.partial" -Destination $Path
    }
}

$root = 'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/52d6c8ffea26cc873ac5ad116f8631268d7eb503'
$venv = Join-Path $work 'venv'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Invoke-Checked 'py.exe' @('-3', '-m', 'venv', $venv)
}
$extract = 'from zipfile import ZipFile; import sys; z=ZipFile(sys.argv[1]); z.extractall(sys.argv[2], ["adapter/adapter_config.json", "adapter/adapter_model.safetensors"])'
Invoke-Checked $python @('-c', $extract, $evidence, $work)

$sourceZip = Join-Path $work 'llama-source.zip'
$runtimeZip = Join-Path $work 'llama-win.zip'
Get-Download 'https://codeload.github.com/ggml-org/llama.cpp/zip/refs/tags/b12800' $sourceZip
Get-Download 'https://github.com/ggml-org/llama.cpp/releases/download/b12800/llama-b12800-bin-win-cpu-x64.zip' $runtimeZip
$source = Join-Path $work 'source'
$runtime = Join-Path $work 'runtime'
Expand-Archive -LiteralPath $sourceZip -DestinationPath $source -Force
Expand-Archive -LiteralPath $runtimeZip -DestinationPath $runtime -Force
$quantizer = (Get-ChildItem -LiteralPath $runtime -Recurse -Filter 'llama-quantize.exe' | Select-Object -First 1).FullName
if (-not $quantizer) { throw 'The Windows archive did not contain llama-quantize.exe.' }
Invoke-Checked $python @('-m', 'pip', 'install', (Join-Path $source 'llama.cpp-b12800\gguf-py'), 'safetensors')
$env:OPENBLAS_NUM_THREADS = '4'
$merger = Join-Path $work 'merge_gguf.py'
Get-Download 'https://raw.githubusercontent.com/mahadbaig2/finetune-anywhere/3eca65e18fdb423a090df2e387dcb671c6989fe5/showcase/qwen-local-hackathon-2026/offline-crop-triage/merge_gguf.py' $merger
$target = Join-Path $ProjectDir 'CropTriage-Q4_K_M.gguf'
if (-not (Test-Path -LiteralPath $target)) {
    $scratch = Join-Path $work ('language-{0}.gguf' -f [guid]::NewGuid())
    try {
        Get-Download "$root/Qwen3VL-2B-Instruct-F16.gguf?download=true" $scratch
        Invoke-Checked $python @($merger, (Join-Path $work 'adapter'), $scratch)
        Invoke-Checked $quantizer @($scratch, "$target.partial", 'Q4_K_M', '4')
        Move-Item -LiteralPath "$target.partial" -Destination $target
    } finally {
        if (Test-Path -LiteralPath $scratch) { Remove-Item -LiteralPath $scratch }
    }
}
$projectorName = 'mmproj-Qwen3VL-2B-Instruct-F16.gguf'
$projector = Join-Path $ProjectDir $projectorName
Get-Download "$root/${projectorName}?download=true" $projector
if ((Get-FileHash -LiteralPath $projector -Algorithm SHA256).Hash -ne 'c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5') {
    throw 'Vision projector checksum failed.'
}
Get-Item -LiteralPath $target, $projector | Select-Object Name,Length,FullName
