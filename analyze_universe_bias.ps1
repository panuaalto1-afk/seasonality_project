# analyze_universe_bias.ps1
# Quick universe bias analysis after backtest completion
# Compares backtest results to SPY/QQQ benchmarks

param(
    [string]$BacktestResultsPath = "seasonality_reports\backtest_results",
    [switch]$Detailed = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BACKTEST UNIVERSE BIAS ANALYSIS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Find latest backtest results folder
Write-Host "[1/5] Finding latest backtest results..." -ForegroundColor Yellow

$latestFolder = Get-ChildItem -Path $BacktestResultsPath -Directory | 
    Where-Object { $_.Name -match "^\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_\d{6}$" } |
    Sort-Object Name -Descending |
    Select-Object -First 1

if (-not $latestFolder) {
    Write-Host "ERROR: No backtest results found in $BacktestResultsPath" -ForegroundColor Red
    exit 1
}

Write-Host "Found: $($latestFolder.Name)" -ForegroundColor Green
Write-Host ""

# Load performance summary
Write-Host "[2/5] Loading performance summary..." -ForegroundColor Yellow

$summaryPath = Join-Path $latestFolder.FullName "performance_summary.txt"

if (-not (Test-Path $summaryPath)) {
    Write-Host "ERROR: performance_summary.txt not found" -ForegroundColor Red
    exit 1
}

$summaryContent = Get-Content $summaryPath -Raw

# Extract key metrics
$totalReturn = 0
if ($summaryContent -match "Total Return:\s+(-?\d+\.?\d*)%") {
    $totalReturn = [double]$Matches[1]
}

$annualReturn = 0
if ($summaryContent -match "Annual Return:\s+(-?\d+\.?\d*)%") {
    $annualReturn = [double]$Matches[1]
}

$sharpe = 0
if ($summaryContent -match "Sharpe Ratio:\s+(-?\d+\.?\d*)") {
    $sharpe = [double]$Matches[1]
}

$maxDD = 0
if ($summaryContent -match "Max Drawdown:\s+-?(-?\d+\.?\d*)%") {
    $maxDD = [double]$Matches[1]
}

# FIXED: Better regex for benchmark returns (multi-line match)
$spyReturn = 0
if ($summaryContent -match "vs SPY:[\s\S]*?Benchmark Return:\s+(-?\d+\.?\d*)%") {
    $spyReturn = [double]$Matches[1]
}

$qqqReturn = 0
if ($summaryContent -match "vs QQQ:[\s\S]*?Benchmark Return:\s+(-?\d+\.?\d*)%") {
    $qqqReturn = [double]$Matches[1]
}

# Extract Alpha and Beta for SPY
$spyAlpha = 0
if ($summaryContent -match "vs SPY:[\s\S]*?Alpha:\s+(-?\d+\.?\d*)%") {
    $spyAlpha = [double]$Matches[1]
}

$spyBeta = 0
if ($summaryContent -match "vs SPY:[\s\S]*?Beta:\s+(-?\d+\.?\d*)") {
    $spyBeta = [double]$Matches[1]
}

Write-Host "Metrics extracted" -ForegroundColor Green
Write-Host ""

# Display backtest results
Write-Host "[3/5] Backtest Performance:" -ForegroundColor Yellow
Write-Host "  Total Return:    $($totalReturn)%" -ForegroundColor White
Write-Host "  Annual Return:   $($annualReturn)%" -ForegroundColor White
Write-Host "  Sharpe Ratio:    $($sharpe)" -ForegroundColor White
Write-Host "  Max Drawdown:    -$($maxDD)%" -ForegroundColor White
Write-Host ""

# Benchmark comparison
Write-Host "[4/5] Benchmark Comparison:" -ForegroundColor Yellow
Write-Host "  SPY:             $($spyReturn)%" -ForegroundColor White
Write-Host "  QQQ:             $($qqqReturn)%" -ForegroundColor White
Write-Host ""

# Calculate outperformance
$spyOutperformance = $totalReturn - $spyReturn
$qqqOutperformance = $totalReturn - $qqqReturn

$spyColor = if ($spyOutperformance -gt 0) { 'Green' } else { 'Red' }
$qqqColor = if ($qqqOutperformance -gt 0) { 'Green' } else { 'Red' }

$spyPrefix = if ($spyOutperformance -gt 0) { '+' } else { '' }
$qqqPrefix = if ($qqqOutperformance -gt 0) { '+' } else { '' }

Write-Host "  vs SPY:          $spyPrefix$($spyOutperformance)%" -ForegroundColor $spyColor
Write-Host "  vs QQQ:          $qqqPrefix$($qqqOutperformance)%" -ForegroundColor $qqqColor
Write-Host ""
Write-Host "  Alpha (vs SPY):  $($spyAlpha)%" -ForegroundColor Cyan
Write-Host "  Beta (vs SPY):   $($spyBeta)" -ForegroundColor Cyan
Write-Host ""

# Universe bias analysis
Write-Host "[5/5] Universe Bias Analysis:" -ForegroundColor Yellow
Write-Host ""

# Determine backtest period
$folderName = $latestFolder.Name
if ($folderName -match "^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})") {
    $startDate = [datetime]::ParseExact($Matches[1], "yyyy-MM-dd", $null)
    $endDate = [datetime]::ParseExact($Matches[2], "yyyy-MM-dd", $null)
    $years = ($endDate - $startDate).Days / 365.25
    
    Write-Host "  Period: $($startDate.ToString('yyyy-MM-dd')) to $($endDate.ToString('yyyy-MM-dd'))" -ForegroundColor White
    Write-Host "  Duration: $([math]::Round($years, 1)) years" -ForegroundColor White
    Write-Host ""
}

# Expected benchmarks (rough estimates)
$expectedSPY_5y = 125  # 2020-2025
$expectedSPY_10y = 225  # 2015-2025

$expectedBenchmark = if ($years -le 6) { $expectedSPY_5y } else { $expectedSPY_10y }

Write-Host "  Expected SPY return (~): $($expectedBenchmark)%" -ForegroundColor Gray
Write-Host "  Actual SPY return:       $($spyReturn)%" -ForegroundColor White
Write-Host ""

# Bias assessment
$excessReturn = $totalReturn - $spyReturn

Write-Host "  ========================================" -ForegroundColor Cyan
Write-Host "  BIAS ASSESSMENT:" -ForegroundColor Cyan
Write-Host "  ========================================" -ForegroundColor Cyan
Write-Host ""

$biasRisk = "LOW"

if ($excessReturn -gt 100) {
    Write-Host "  WARNING: HIGH OUTPERFORMANCE (+$([math]::Round($excessReturn,1))%)" -ForegroundColor Yellow
    Write-Host "  -> Possible survivorship bias" -ForegroundColor Yellow
    Write-Host "  -> Add 10-20% safety margin to expectations" -ForegroundColor Yellow
    $biasRisk = "HIGH"
}
elseif ($excessReturn -gt 50) {
    Write-Host "  WARNING: MODERATE OUTPERFORMANCE (+$([math]::Round($excessReturn,1))%)" -ForegroundColor Yellow
    Write-Host "  -> Some survivorship bias likely" -ForegroundColor Yellow
    Write-Host "  -> Add 5-10% safety margin to expectations" -ForegroundColor Yellow
    $biasRisk = "MODERATE"
}
elseif ($excessReturn -gt 0) {
    Write-Host "  OK: REASONABLE OUTPERFORMANCE (+$([math]::Round($excessReturn,1))%)" -ForegroundColor Green
    Write-Host "  -> Universe bias likely minimal" -ForegroundColor Green
    Write-Host "  -> Results appear realistic" -ForegroundColor Green
    Write-Host "  -> Low Beta ($spyBeta) suggests market-neutral strategy" -ForegroundColor Green
    $biasRisk = "LOW"
}
else {
    Write-Host "  WARNING: UNDERPERFORMANCE ($([math]::Round($excessReturn,1))%)" -ForegroundColor Red
    Write-Host "  -> System may not be working as expected" -ForegroundColor Red
    Write-Host "  -> Review strategy parameters" -ForegroundColor Red
    $biasRisk = "N/A"
}

Write-Host ""

# Adjusted expectations
if ($biasRisk -eq "HIGH") {
    $adjustedReturn = $totalReturn * 0.85  # -15%
    Write-Host "  ADJUSTED EXPECTATION (conservative):" -ForegroundColor Cyan
    Write-Host "    $([math]::Round($adjustedReturn, 1))% total return" -ForegroundColor White
    Write-Host "    (accounting for potential 15% survivorship bias)" -ForegroundColor Gray
}
elseif ($biasRisk -eq "MODERATE") {
    $adjustedReturn = $totalReturn * 0.92  # -8%
    Write-Host "  ADJUSTED EXPECTATION (moderate):" -ForegroundColor Cyan
    Write-Host "    $([math]::Round($adjustedReturn, 1))% total return" -ForegroundColor White
    Write-Host "    (accounting for potential 8% survivorship bias)" -ForegroundColor Gray
}
else {
    Write-Host "  No adjustment needed - results appear realistic" -ForegroundColor Green
    Write-Host "  Deploy confidence: HIGH" -ForegroundColor Green
}

Write-Host ""

# Detailed analysis (if requested)
if ($Detailed) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "DETAILED ANALYSIS" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Load trades history
    $tradesPath = Join-Path $latestFolder.FullName "trades_history.csv"
    
    if (Test-Path $tradesPath) {
        Write-Host "Loading trades history..." -ForegroundColor Yellow
        
        $trades = Import-Csv $tradesPath
        
        $totalTrades = $trades.Count
        $winningTrades = ($trades | Where-Object { [double]$_.pl_pct -gt 0 }).Count
        $winRate = ($winningTrades / $totalTrades) * 100
        
        $avgHoldDays = ($trades | Measure-Object -Property hold_days -Average).Average
        
        Write-Host ""
        Write-Host "Trade Statistics:" -ForegroundColor White
        Write-Host "  Total Trades:    $totalTrades" -ForegroundColor White
        Write-Host "  Winning Trades:  $winningTrades" -ForegroundColor White
        Write-Host "  Win Rate:        $([math]::Round($winRate, 1))%" -ForegroundColor White
        Write-Host "  Avg Hold Days:   $([math]::Round($avgHoldDays, 1))" -ForegroundColor White
        Write-Host ""
        
        # Check for outliers (huge winners)
        $bigWinners = $trades | Where-Object { [double]$_.pl_pct -gt 50 }
        
        if ($bigWinners.Count -gt 0) {
            Write-Host "WARNING: Big Winners (>+50% each):" -ForegroundColor Yellow
            $bigWinners | Select-Object -First 10 | ForEach-Object {
                $tickerInfo = "$($_.ticker): +$($_.pl_pct)% - Entry: $($_.entry_date), Exit: $($_.exit_date)"
                Write-Host "    $tickerInfo" -ForegroundColor Gray
            }
            if ($bigWinners.Count -gt 10) {
                Write-Host "    ... and $($bigWinners.Count - 10) more" -ForegroundColor Gray
            }
            Write-Host ""
            Write-Host "  -> These may be survivorship bias candidates" -ForegroundColor Yellow
            Write-Host "  -> Validate if these stocks were 'known winners' at entry" -ForegroundColor Yellow
        }
        else {
            Write-Host "  No extreme outliers found (all trades <+50%)" -ForegroundColor Green
            Write-Host "  -> Suggests realistic, consistent strategy" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ANALYSIS COMPLETE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create summary file
$spyOutPrefixFile = if ($spyOutperformance -gt 0) { '+' } else { '' }
$qqqOutPrefixFile = if ($qqqOutperformance -gt 0) { '+' } else { '' }

$analysisOutput = @"
UNIVERSE BIAS ANALYSIS
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

BACKTEST PERFORMANCE:
  Total Return:    $($totalReturn)%
  Annual Return:   $($annualReturn)%
  Sharpe Ratio:    $($sharpe)
  Max Drawdown:    -$($maxDD)%

BENCHMARKS:
  SPY:             $($spyReturn)%
  QQQ:             $($qqqReturn)%
  
  vs SPY:          $spyOutPrefixFile$([math]::Round($spyOutperformance,2))%
  vs QQQ:          $qqqOutPrefixFile$([math]::Round($qqqOutperformance,2))%
  
  Alpha (vs SPY):  $($spyAlpha)%
  Beta (vs SPY):   $($spyBeta)

BIAS ASSESSMENT:
  Risk Level:      $biasRisk
  Recommendation:  $(if ($biasRisk -eq "HIGH") { "Add 10-20% safety margin" } elseif ($biasRisk -eq "MODERATE") { "Add 5-10% safety margin" } else { "Results appear realistic - Deploy with confidence" })

INTERPRETATION:
  $(if ($biasRisk -eq "LOW") { "Low Beta ($spyBeta) and reasonable outperformance (+$([math]::Round($excessReturn,1))%) suggest a market-neutral strategy with minimal survivorship bias. Results are likely realistic." } elseif ($biasRisk -eq "MODERATE") { "Moderate outperformance may indicate some survivorship bias. Consider validating with point-in-time universe." } else { "High outperformance suggests significant survivorship bias. Recommend fundamental validation or SP500-universe backtest." })

"@

$analysisPath = Join-Path $latestFolder.FullName "universe_bias_analysis.txt"
$analysisOutput | Out-File -FilePath $analysisPath -Encoding UTF8

Write-Host "Analysis saved to: $analysisPath" -ForegroundColor Green
Write-Host ""