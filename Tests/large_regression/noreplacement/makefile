CXX = g++
CXXFLAGS = -std=c++11 -O3
LIBS = -fopenmp -lm -I ./../../
LIBFLAGS = -DEIGEN_DONT_PARALLELIZE 
DEBUG = -Wall -Werror

all: $(patsubst %.cc, %.out, $(wildcard *.cc))
%.out: %.cc makefile
	$(CXX) $(CXXFLAGS) $(LIBS) $(LIBFLAGS) $< -o $@
clean:
	rm -f *.out
remake:
	make clean
	make -j

.PHONY: all, clean
