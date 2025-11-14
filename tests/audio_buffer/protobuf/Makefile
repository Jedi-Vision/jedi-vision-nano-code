CC = gcc
CFLAGS = -Wall -Wextra -O2 -lczmq -I/opt/homebrew/include -L/opt/homebrew/lib -I pbtools/lib/include
LDFLAGS = 
TARGET = server.out
SRCS = server.c pbtools/lib/src/pbtools.c ../../../src/jv/pb/objectrep.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean