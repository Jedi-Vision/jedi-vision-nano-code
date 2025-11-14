/**
 * Request-and-reply server example code using ZeroMQ.
 */

#include <czmq.h>

typedef struct {
    int x;
} Data;


int main(int argc, char const *argv[])
{
    //  Socket to talk to clients
    zsock_t *responder = zsock_new(ZMQ_REP);
    int rc = zsock_bind(responder, "tcp://*:5555");
    assert(rc == 5555);

    while (1) {
        char *buffer = zstr_recv(responder);

        // Copy bytes into struct
        Data data;
        memcpy(&data, buffer, sizeof(Data));
        printf("x=%d\n", data.x);

        sleep (1);          //  Do some 'work'
        zstr_send(responder, "0");
        zstr_free(&buffer);
    }
    return 0;
}