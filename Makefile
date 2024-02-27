.PHONY: lint test

lint:
	black .
	isort .

test:
	python -m unittest


