.PHONY: run test clean tags

run:
	pipenv run python -m src.main

test:
	pipenv run python -m unittest discover

typecheck:
	pipenv run mypy -p src -p test

format:	isort	black
	
black:	
	find . -name '*.py' -a \! -path '*/\.*' | xargs black

isort:
	pipenv run isort --skip-glob=.tox --recursive .

clean:
	rm --recursive --force ./tags && \
		rm --force --recursive .mypy_cache/

tags:
	rm --force ./tags && \
		ctags -R --exclude=data/interim --exclude=.git --exclude=.mypy_cache --exclude=.undodir .
