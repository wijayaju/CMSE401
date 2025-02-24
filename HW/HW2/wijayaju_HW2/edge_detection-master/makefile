all: process 

main_process.o: main_process.c
	gcc -c main_process.c 
pt2_main_process.o: pt2_main_process.c
	gcc -c -O2 pt2_main_process.c
pt3_main_process.o: pt3_main_process.c
	gcc -c -fopenmp pt3_main_process.c

png_util.o: png_util.c
	gcc -l lpng16 -c png_util.c

process: main_process.o png_util.o
	gcc -o process -lm -l png16 main_process.o png_util.o
process2: pt2_main_process.o png_util.o
	gcc -o process2 -lm -l png16 pt2_main_process.o png_util.o
process3: pt3_main_process.o png_util.o
	gcc -o process3 -fopenmp -lm -l png16 pt3_main_process.o png_util.o

test: process process2 process3
	./process ./images/cube.png test.png
	./process2 ./images/cube.png test.png
	./process3 ./images/cube.png test.png

clean:
	rm *.o
	rm process 
