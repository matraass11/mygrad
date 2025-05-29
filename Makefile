version = -std=c++14

main: main.o tensor.o
	clang++ main.o tensor.o -o main

main.o: main.cpp
	clang -c $(version) -g main.cpp

tensor.o: tensor.cpp
	clang -c $(version) -g tensor.cpp 
	
clean:
	rm *.o
	
# maind: 
# 	clang++ $(version) -o main.o main.cpp -g
