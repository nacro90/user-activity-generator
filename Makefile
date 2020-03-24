unittest:
	python -m unittest discover

typecheck:
	mypy -p src -p test

format:	isort
	find . -name '*.py' -a \! -path '*/\.*' | xargs black

isort:
	isort --skip-glob=.tox --recursive .

clean-mypy:
	rm --force --recursive .mypy_cache/

