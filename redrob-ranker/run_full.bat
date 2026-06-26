@echo off
echo Starting Precomputation on 100k candidates...
python "%~dp0scripts\precompute.py" --candidates "%~dp0..\candidates.jsonl"
if %errorlevel% neq 0 exit /b %errorlevel%

echo Starting Ranking Pipeline on 100k candidates...
python "%~dp0rank.py" --candidates "%~dp0..\candidates.jsonl" --out "%~dp0..\final_submission.csv"
if %errorlevel% neq 0 exit /b %errorlevel%

echo All done! Pipeline successful.
