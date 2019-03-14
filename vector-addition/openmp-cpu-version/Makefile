CC=gcc
CFLAGS=-O3 -fopenmp

vec_add : vector_add.c
	$(CC) -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f vec_add