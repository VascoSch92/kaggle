# Minimal Makefile for development

# variables
# you can set the first variable from the environment
PYPI_PASSWORD   ?=
DISTDIR         = dist
DOCSDIR         = docs
SOURCEDIR       = symmetria
TESTSDIR        = tests

# It is first such that "make" without argument is like "make help".
help:
	@echo "[HELP] Makefile commands:"
	@echo " * clean: clean caches and others."
	@echo " * ruff: execute ruff check and format."


.PHONY: help Makefile

clean:
	@echo "[INFO] Clean caches and others"
	@rm -rf "./$(DISTDIR)"
	@rm -rf "./.pytest_cache"
	@rm -rf "./.cache"
	@rm -rf "./catboost_info"

ruff:
	@echo "[INFO] Ruff checks & format"
	@ruff format .
	@ruff check . --fix

