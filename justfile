export JUST_LOG := "warn"

watch:
    git ls-files | entr -r python3 app.py

run:
    python3 app.py
