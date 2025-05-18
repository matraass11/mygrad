.PHONY : all main clean
run: main
	./main.o

main: 
	clang++ -std=c++11 main.cpp -o main.o

debug: maind
	./main.o

maind: 
	clang++ -std=c++11 -o main.o main.cpp -g
