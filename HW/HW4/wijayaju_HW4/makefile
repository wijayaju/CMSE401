all: revGOL revGOL-basic_mpi revGOL-mpi gol 

CC:=mpicc
EXT:=c
FLAGS:= -lm -lpng

reverseGOL.o: reverseGOL.$(EXT)
	$(CC) $(FLAGS) -c reverseGOL.$(EXT)

reverseGOL-basic_mpi.o: reverseGOL-basic_mpi.$(EXT)
	$(CC) $(FLAGS) -c reverseGOL-basic_mpi.$(EXT)

reverseGOL-mpi.o: reverseGOL-mpi.$(EXT)
	$(CC) $(FLAGS) -c reverseGOL-mpi.$(EXT) 

png_util.o: png_util.c
	$(CC) $(FLAGS) -c png_util.c

revGOL: reverseGOL.o png_util.o
	$(CC) $(FLAGS) -o revGOL reverseGOL.o png_util.o

revGOL-basic_mpi: reverseGOL-basic_mpi.o png_util.o
	$(CC) $(FLAGS) -o revGOL-basic_mpi reverseGOL-basic_mpi.o png_util.o

revGOL-mpi: reverseGOL-mpi.o png_util.o
	$(CC) $(FLAGS) -o revGOL-mpi reverseGOL-mpi.o png_util.o

gameoflife.o: gameoflife.$(EXT)
	$(CC) $(FLAGS) -c gameoflife.$(EXT) 

gol: gameoflife.o png_util.o
	$(CC) $(FLAGS) -o gol gameoflife.o png_util.o

test: revGOL revGOL-mpi data.txt
	./revGOL data.txt
	mpiexec -np 1 ./revGOL-basic_mpi data.txt
	mpiexec -np 4 ./revGOL-mpi data.txt

clean:
	rm *.o
	rm gol 
	rm revGOL
	rm revGOL-basic_mpi
	rm revGOL-mpi 
