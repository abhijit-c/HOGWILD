CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native
LIBS = -fopenmp -I ./../../
LIBFLAGS = -DEIGEN_DONT_PARALLELIZE 
DEBUG = -Wall -Werror

all: $(patsubst %.cc, %.out, $(wildcard *.cc))
%.out: %.cc makefile
	$(CXX) $(CXXFLAGS) $(LIBS) $(LIBFLAGS) $< -o $@ -lm
clean:
	rm -f *.out
remake:
	make clean
	make -j

.PHONY: all, clean
