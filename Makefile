all:
	mpicc proj.c -o proj -lm
test: test.c proj.h
	mpicc test.c -o test -lm
clean:
	rm *.btr