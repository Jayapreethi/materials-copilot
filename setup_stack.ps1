param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardedArgs
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
    $activateScript = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        . $activateScript
    }

    python .\setup_stack.py @ForwardedArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}