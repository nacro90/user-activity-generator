unittest:
	pipenv run python -m unittest discover

typecheck:
	pipenv run mypy -p src -p test

format:
	find . -name '*.py' -a \! -path '*/\.*' | xargs black

isort:
	isort --skip-glob=.tox --recursive .

clean-mypy:
	rm --force --recursive .mypy_cache/

run:
	pipenv run python -m src.main
