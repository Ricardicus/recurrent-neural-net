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

cpplint:
ifeq (, which cpplint)
	@echo "install cpplint before attempting to make check"
else
	cpplint \
--filter=-build/include_subdir,-build/header_guard,-legal/copyright,-build/include_what_you_use,\
-readability/casting,-build/include_subdir,-build/include_order,-readability/multiline_string \
--linelength=130 \
--verbose=5 --recursive src

endif

