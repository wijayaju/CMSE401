all: pi.o

pi.o: pi.c
	mpicc pi.c -o pi.o

test:
	mpiexec -np 2 ./pi.o 

clean:
	rm pi.o
