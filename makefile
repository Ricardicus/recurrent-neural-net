CC := gcc
FLAGS := O3 Ofast msse3 Wall
LIB := m
SOURCES := *.c
.PHONY : net

all: net

net:
	$(CC) $(SOURCES) $(addprefix -, $(FLAGS)) $(addprefix -l, $(LIB)) -o $@


