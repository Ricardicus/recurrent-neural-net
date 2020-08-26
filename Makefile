.PHONY: docs net clean

all: net

net:
	@make -C src
	@cp src/net .

clean:
	@make $@ -C src

docs:
ifeq (,which doxygen)
	@echo "In order to build the docs, doxygen needs to be installed"
else
	@cd docs;doxygen doxygenConfig;cd ..
endif
