param(
    [Parameter(Mandatory=$true)][string]$SilverDir,
    [Parameter(Mandatory=$true)][string]$GoldDir,
    [Parameter(Mandatory=$true)][string]$SenseRepo,
    [string]$Output = "data/processed/silver_pairs.jsonl",
    [string]$CoverageOutput = "outputs/coverage_errors.jsonl"
)
lexisense-distill build-dataset --silver-dir $SilverDir --gold-dir $GoldDir --sense-repo $SenseRepo --out $Output --coverage-out $CoverageOutput
exit $LASTEXITCODE
