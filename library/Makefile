# Script arguments

PIP_CACHE_FILES := $(shell find . -name '*.so' -o -wholename '*/__pycache__' -o -wholename '*.egg-info' -o -wholename './build' -o -wholename '*.pytest_cache')

.PHONY: clean
clean:
	@rm -rf $(PIP_CACHE_FILES)
	