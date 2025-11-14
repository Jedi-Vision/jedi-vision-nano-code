CC = cc
CFLAGS = -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/lib
LIBS = -lczmq

SRC = example_server.c
TARGET = server.out

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)