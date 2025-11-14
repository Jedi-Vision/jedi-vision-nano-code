CC = CC
CFLAGS = -Wall -Wextra -O2 -lczmq -I/opt/homebrew/include -L/opt/homebrew/lib
LDFLAGS = 
TARGET = server.out
SRCS = server.c \
	../../../src/c_utils/c_utils.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean