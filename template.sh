cd "$(dirname "$0")"

# Creating directory 
mkdir -p research
mkdir -p config
mkdir -p packages/core/src/core

# Creating files
touch research/eda.ipynb

touch packages/core/pyproject.toml
touch packages/core/src/core/__init__.py
touch packages/core/src/core/logger.py
touch packages/core/src/core/exception.py

touch .gitignore
touch .env
touch requirements.txt
touch README.md

touch app.py
touch config/config.yaml
touch params.yaml
touch pyproject.toml


echo "Directory and files created successfully!."