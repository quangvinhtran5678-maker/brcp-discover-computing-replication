param(
    [string]$Out = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([string]::IsNullOrWhiteSpace($Out)) {
    $Out = Join-Path $projectRoot "data\Class-1"
}
$outRoot = [System.IO.Path]::GetFullPath($Out)

Write-Host "BR-CP official dataset downloader"
Write-Host "Output: $outRoot"
Write-Host ""

$dirs = @(
    "ETT-small",
    "electricity",
    "traffic",
    "weather",
    "exchange_rate",
    "illness"
)

foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $outRoot $d) | Out-Null
}

$files = @(
    @("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh1.csv?download=true", "ETT-small\ETTh1.csv"),
    @("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh2.csv?download=true", "ETT-small\ETTh2.csv"),
    @("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm1.csv?download=true", "ETT-small\ETTm1.csv"),
    @("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm2.csv?download=true", "ETT-small\ETTm2.csv"),
    @("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv?download=true", "electricity\electricity.csv"),
    @("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv?download=true", "traffic\traffic.csv"),
    @("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv?download=true", "weather\weather.csv"),
    @("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/exchange_rate/exchange_rate.csv?download=true", "exchange_rate\exchange_rate.csv"),
    @("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/illness/national_illness.csv?download=true", "illness\national_illness.csv")
)

$i = 0
foreach ($f in $files) {
    $i++
    $url = $f[0]
    $rel = $f[1]
    $dest = Join-Path $outRoot $rel
    Write-Host "[$i/9] Downloading $rel ..."
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    if (-not (Test-Path $dest)) {
        throw "Download failed: $rel"
    }
    $sizeMB = [math]::Round((Get-Item $dest).Length / 1MB, 2)
    Write-Host "      OK - $sizeMB MB"
}

Write-Host ""
Write-Host "=== DOWNLOAD COMPLETE ==="
Write-Host ""
Get-ChildItem $outRoot -Recurse -File |
    Select-Object FullName, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}} |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Folder ready:"
Write-Host $outRoot
Write-Host ""
Write-Host "Set BRCP_DATA to this directory when running the real-data scripts."
