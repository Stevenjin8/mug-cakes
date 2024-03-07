.PHONY: lint test coverage-test coverage-html

lint:
	black .
	isort .

test:
	python -m unittest

test-coverage:
	coverage run -m unittest

coverage-html:
	coverage html

