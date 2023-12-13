CC=g++
CFLAGS=-std=c++20 -Wall -Wextra

all: dijkstra

dijkstra: dijkstra.o
	$(CC) -O0 -o $@ $^

dijkstra.o: dijkstra.cpp
	$(CC) -O0 -o $@ -c $< $(CFLAGS)


.PHONY: clean
clean:
	rm -f dijkstra