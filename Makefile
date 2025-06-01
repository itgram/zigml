# silence any make output to avoid interfere with manifests apply
.SILENT:

# Delete the default suffixes
.SUFFIXES:

# sets the default goal to be used if no targets were specified on the command line
.DEFAULT_GOAL := default

.PHONY: default
default: run ## run the application

.PHONY: build
build: ## build the application
	zig build 

.PHONY: run
run: ## run the application
	zig build run 

.PHONY: test
test: ## run all unit tests
	zig build test --summary all

.PHONY: help
help: ## display available commands
	$(info make targets:)
	printf "\n"
	awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[\/a-zA-Z\-\_0-9%:\\]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
