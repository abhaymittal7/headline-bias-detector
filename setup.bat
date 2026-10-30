@echo off
REM One-time setup for Windows. Run from the project folder: setup.bat
cd /d "%~dp0"

set "PY=python"
py -3.12 -c "" >nul 2>nul && set "PY=py -3.12"
echo Using: %PY%

if not exist venv (
    %PY% -m venv venv || goto :fail
)
call venv\Scripts\activate.bat || goto :fail

python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :fail
pip install -r requirements.txt || goto :fail
python -m spacy download en_core_web_md || goto :fail
python -c "import nltk; nltk.download('vader_lexicon')" || goto :fail
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || goto :fail

echo.
echo Setup complete. Now run: run.bat
goto :eof

:fail
echo.
echo Setup FAILED at the step above. Read the error message.
exit /b 1
