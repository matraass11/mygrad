version=-std=c++14

main: main.o tensor.o functions.o helper.o
	clang++ $^ -o main

$.o: $.cpp $.hpp
	clang++ -c $(version) -g $^

clean:
	rm *.o
	
# maind: 
# 	clang++ $(version) -o main.o main.cpp -g
