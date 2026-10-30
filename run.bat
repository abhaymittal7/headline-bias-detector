@echo off
REM Windows launcher. Usage:
REM   run.bat                      full pipeline (needs internet for RSS)
REM   run.bat collect              grow dataset to 5000 headlines (needs internet)
REM   run.bat evaluate             accuracy test on expert-labelled BABE data (offline)
REM   run.bat stats                dataset size per source (offline)
REM   run.bat report               report only, from the existing database
REM   run.bat headline             interactive single-headline analysis (offline)
REM   run.bat headline "Some text" analyze one headline
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
    echo venv not found. Run setup.bat first.
    exit /b 1
)
call venv\Scripts\activate.bat

if /i "%~1"=="evaluate" (
    python -m scripts.evaluate
) else if /i "%~1"=="collect" (
    python -m scripts.collect_headlines %2
) else if /i "%~1"=="stats" (
    python -m scripts.collect_headlines --stats
) else if /i "%~1"=="report" (
    python -m src.analysis.reporting
) else if /i "%~1"=="headline" (
    python -m scripts.analyze_headline %2 %3 %4 %5 %6 %7 %8 %9
) else (
    python main.py
)
