run: main
	./main.o

main: 
	clang++ -o main.o main.cpp

debug: maind
	./main.o

maind: 
	clang++ -o main.o main.cpp -g
