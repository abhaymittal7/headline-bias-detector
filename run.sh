#!/usr/bin/env bash
# Linux/macOS launcher.
#   ./run.sh setup              one-time install (venv, deps, models)
#   ./run.sh                    full pipeline (needs internet for RSS)
#   ./run.sh report             report only, from the existing database
#   ./run.sh headline           interactive single-headline analysis (offline)
#   ./run.sh headline "Text"    analyze one headline
set -e
cd "$(dirname "$0")"

if [ "$1" = "setup" ]; then
    PY=python3
    command -v python3.12 >/dev/null 2>&1 && PY=python3.12
    echo "Using: $PY"
    [ -d venv ] || "$PY" -m venv venv
    . venv/bin/activate
    python -m pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    python -m spacy download en_core_web_md
    python -c "import nltk; nltk.download('vader_lexicon')"
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
    echo "Setup complete. Now run: ./run.sh"
    exit 0
fi

if [ ! -f venv/bin/activate ]; then
    echo "venv not found. Run: ./run.sh setup"
    exit 1
fi
. venv/bin/activate

case "$1" in
    evaluate) python -m scripts.evaluate ;;
    collect)  shift; python -m scripts.collect_headlines "$@" ;;
    stats)    python -m scripts.collect_headlines --stats ;;
    report)   python -m src.analysis.reporting ;;
    headline) shift; python -m scripts.analyze_headline "$@" ;;
    *)        python main.py ;;
esac
