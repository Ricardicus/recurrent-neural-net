CC := gcc
FLAGS := O3 Ofast msse3
LIBS := m
GCC_HINTS := all \
  unused \
  uninitialized \
  no-unused-variable \
  extra \
  unused-parameter

.PHONY : net clean

SRCS := layers.c \
		lstm.c \
		main.c \
		set.c \
		utilities.c

OBJS := $(subst .c,.o,$(SRCS))

all: net

%.o : %.c
	$(CC) -c $< $(addprefix -, $(FLAGS)) $(addprefix -W, $(GCC_HINTS)) \
		$(addprefix -l, $(LIBS)) -o $@

net: $(OBJS)
	$(CC) $^ $(addprefix -, $(FLAGS)) $(addprefix -W, $(GCC_HINTS)) \
		$(addprefix -l, $(LIBS)) -o $@

clean:
	rm -f $(OBJS)

