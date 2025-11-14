/**
 * Request-and-reply server example code using ZeroMQ.
 */

#include <czmq.h>
#include "../../../src/c_utils/c_utils.h"

typedef struct {
    int x;
} Data;


int main(int argc, char const *argv[])
{
    // Check if the directory for ipc_path exists, if not create it
    char *ipc_path = "ipc:///tmp/jv/audio/0.sock";
    char *dir_path = "/tmp/jv/audio/";
    _mkdir(dir_path, 0777);

    //  Socket to talk to clients
    zsock_t *responder = zsock_new(ZMQ_REP);
    int rc = zsock_bind(responder, "%s", ipc_path);
    assert(rc == 0);

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