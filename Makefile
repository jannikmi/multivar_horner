lock:
	poetry lock

update:
	@echo "pinning the dependencies specified in 'pyproject.toml':"
	poetry update

install:
	poetry install --with docs,dev --all-extras --sync

VENV_NAME=multivar_horner
venv:
	conda create -y -n ${VENV_NAME} python=3.8 poetry
	(conda activate ${VENV_NAME} && poetry install)

hook:
	pre-commit install
	pre-commit run --all-files

hookup:
	pre-commit autoupdate

test:
	poetry run pytest

tox:
	tox


clean:
	rm -rf .pytest_cache .coverage coverage.xml tests/__pycache__ src/__pycache__ mlruns/ .mypyp_cache/

# documentation generation:
# https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html
docs:
	(cd docs && make html)


.PHONY: clean test venv vpn docs
