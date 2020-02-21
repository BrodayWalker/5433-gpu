SRC = ./
INC = ./
OBJ = fft.o

fft : $(OBJ)
	gcc -o fft.exe $(OBJ)

fft.o : WalkerBrodaySeqVer.c 
	gcc -c WalkerBrodaySeqVer.c -std=c99 -o fft.o

clean :
	rm *.exe fftJobScript.* $(OBJ)