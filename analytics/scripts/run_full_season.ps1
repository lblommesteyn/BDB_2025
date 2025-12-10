# Run the full season batch processing
# This may take 4-5 hours depending on your hardware

$ErrorActionPreference = "Stop"

Write-Host "Starting full season batch processing..."
Write-Host "This process will analyze approximately 270 games."

# Define paths
$RepoRoot = "c:\Users\16476\BDB_2025"
$SummaryPath = "$RepoRoot\season_summary.parquet"
$OutDir = "$RepoRoot\analytics\outputs\dacs_final_full"
$ReportDir = "$RepoRoot\analytics\outputs\report_full"

# 1. Run Batch Runner
# Removing --limit to process all games
# Using --no-outcome-model for now as we haven't trained it yet
Write-Host "Step 1: Running Batch Runner..."
python "$RepoRoot\analytics\batch_runner.py" `
    --season-summary $SummaryPath `
    --out $OutDir `
    --uncertainty_samples 50

Write-Host "Batch processing complete. Summary saved to $SummaryPath"

# 2. Generate Analysis Report
Write-Host "Step 2: Generating Analysis Report..."
python "$RepoRoot\analytics\generate_analysis_report.py" `
    --summary $SummaryPath `
    --out $ReportDir

Write-Host "Full season analysis complete!"
Write-Host "Reports are available in: $ReportDir"
