unittest:
	pipenv run python -m unittest discover

typecheck:
	pipenv run mypy -p src -p test

format:	isort	black
	
black:	
	find . -name '*.py' -a \! -path '*/\.*' | xargs black

isort:
	pipenv run isort --skip-glob=.tox --recursive .

clean-mypy:
	rm --force --recursive .mypy_cache/

run:
	pipenv run python -m src.main
