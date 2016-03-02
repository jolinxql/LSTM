CC = g++
OBJ = utils.o Net.o
CFLAGS = -I /home/xql/eigen/ -O2 -funroll-loops -Wno-unused-result
TARGET = LstmTrainTest

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) main.cpp

$(OBJ): %.o:%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

all: $(TARGET)
clean:
	@rm -rf *.o $(TARGET)
