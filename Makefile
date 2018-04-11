CC = nvcc
BIN = bin/test
OBJ = obj/test.o
SRC = src/test.cu

all:$(BIN)

$(BIN):$(OBJ)
	$(CC) $(OBJ) -o $(BIN)

$(OBJ):$(SRC)
	$(CC) -c $(SRC) -o $(OBJ)

clean:
	-rm $(BIN) $(OBJ)
