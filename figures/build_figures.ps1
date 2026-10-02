# Build all figures into figures/ (pdf + png).
# pdflatex is run inside a scratch directory because the sandbox blocks direct
# writes from child processes inside the workspace.
$ErrorActionPreference = 'Stop'
$figdir = 'E:\paper-tianjinUnivercity\figures'
$build  = Join-Path $env:TEMP 'figbuild'
$pdftoppm = 'C:\Users\yangkai\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe'

if (Test-Path $build) { Remove-Item $build -Recurse -Force }
New-Item -ItemType Directory -Force -Path $build, (Join-Path $build 'data') | Out-Null
Copy-Item (Join-Path $figdir '*.tex') $build -Force
Copy-Item (Join-Path $figdir 'data\*') (Join-Path $build 'data') -Force

Push-Location $build
try {
  foreach ($tex in (Get-ChildItem $build -Filter *.tex)) {
    $base = [IO.Path]::GetFileNameWithoutExtension($tex.Name)
    Write-Host "== $base =="
    & pdflatex -interaction=nonstopmode -halt-on-error $tex.Name | Out-Null
    if (-not (Test-Path (Join-Path $build "$base.pdf"))) { throw "pdflatex failed for $base" }
    & $pdftoppm -png -r 130 (Join-Path $build "$base.pdf") (Join-Path $build $base)
    foreach ($f in @("$base.pdf", "$base-1.png", "$base.log")) {
      $src = Join-Path $build $f
      if (Test-Path $src) { Copy-Item $src $figdir -Force }
    }
    $png = Join-Path $figdir "$base-1.png"
    if (Test-Path $png) { Move-Item $png (Join-Path $figdir "$base.png") -Force }
    Write-Host "   -> $base.pdf / $base.png"
  }
} finally { Pop-Location }
Get-ChildItem (Join-Path $figdir '*.pdf'), (Join-Path $figdir '*.png') |
  Select-Object Name, Length | Format-Table -AutoSize
