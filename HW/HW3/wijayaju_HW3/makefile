all: gol0 gol1 gol2 

CC:=gcc
EXT:=c
FLAGS:=-lm -lpng16

CC2:=nvcc
EXT2:=cu

gameoflife0.o: gameoflife0.$(EXT)
	$(CC) $(FLAGS) -c gameoflife0.$(EXT) 

png_util0.o: png_util0.$(EXT)
	$(CC) $(FLAGS) -c png_util0.$(EXT)

gol0: gameoflife0.o png_util0.o
	$(CC) $(FLAGS) -o gol0 gameoflife0.o png_util0.o


gameoflife1.o: gameoflife1.$(EXT)
	$(CC) $(FLAGS) -c -O3 gameoflife1.$(EXT) 

png_util1.o: png_util1.$(EXT)
	$(CC) $(FLAGS) -c png_util1.$(EXT)

gol1: gameoflife1.o png_util1.o
	$(CC) $(FLAGS) -o gol1 gameoflife1.o png_util1.o


gameoflife2.o: gameoflife2.$(EXT2)
	$(CC2) $(FLAGS) -c -diag-suppress 2464 gameoflife2.$(EXT2) 

png_util2.o: png_util2.$(EXT2)
	$(CC2) $(FLAGS) -c png_util2.$(EXT2)

gol2: gameoflife2.o png_util2.o
	$(CC2) $(FLAGS) -o gol2 gameoflife2.o png_util2.o


test: gol0 gol1 gol2 data.txt
	./gol0 < data.txt
	./gol1 < data.txt
	sbatch jobScript.sb
	echo "Please check most recent slurm file for CUDA version outputs."

test2: gol2 data.txt
	cuda-memcheck ./gol2 < data.txt

test3: gol0 gol1 gol2
	echo "0 5" | time ./gol0
	echo "0 5" | time ./gol1
	sbatch jobScriptScalingTest.sb
	echo "Please check most recent slurm file for CUDA version outputs."
clean:
	rm *.o
	rm gol* 

