all:
	gcc *.c -O3 -Ofast -msse3 -lm -o net
