run: grad
	./grad.o

grad: 
	clang++ -o grad.o grad.cpp

