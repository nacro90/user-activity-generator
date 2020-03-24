unittest:
	python -m unittest discover

typecheck:
	mypy -p src -p test

format:	isort	black
	
black:	
	find . -name '*.py' -a \! -path '*/\.*' | xargs black

isort:
	isort --skip-glob=.tox --recursive .

clean-mypy:
	rm --force --recursive .mypy_cache/

