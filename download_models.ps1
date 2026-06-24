Write-Host "Starting EasyOCR model download from Hugging Face..."

$modelDir = "C:\Users\Acer\.EasyOCR\model"

# Create model directory if not exist
if (!(Test-Path -Path $modelDir)) {
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
}

$files = @(
    @{
        Url = "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/craft_mlt_25k.pth"
        Path = "$modelDir\craft_mlt_25k.pth"
    },
    @{
        Url = "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/cyrillic_g2.pth"
        Path = "$modelDir\cyrillic_g2.pth"
    },
    @{
        Url = "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/latin_g2.pth"
        Path = "$modelDir\latin_g2.pth"
    }
)

foreach ($file in $files) {
    if (Test-Path -Path $file.Path) {
        Write-Host "Model file already exists: $($file.Path)"
        continue
    }

    Write-Host "Downloading $($file.Url)..."
    # Download directly using curl.exe
    curl.exe -L -y 30 -Y 1000 --connect-timeout 60 -o $file.Path $file.Url

    if (Test-Path -Path $file.Path) {
        Write-Host "Success: $($file.Path) downloaded successfully."
    } else {
        Write-Error "Failed to download from $($file.Url)"
    }
}

Write-Host "Model setup script finished."
