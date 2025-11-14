/**
 * Request-and-reply server example code using ZeroMQ.
 */

#include <czmq.h>
#include <stdlib.h>
#include "../../../c_utils/c_utils.h"


int main(int argc, char const *argv[])
{

    // Check if the directory for ipc_path exists, if not create it
    const char *ipc_path = "ipc:///tmp/jv/audio/0";
    char *dir_path = "/tmp/jv/audio";
    _mkdir(dir_path, 0777);

    //  Socket to talk to clients
    zsock_t *responder = zsock_new(ZMQ_REP);
    int rc = zsock_bind(responder, "%s", ipc_path);
    assert(rc == 0);

    while (1) {
        char *str = zstr_recv(responder);
        printf("%s\n", str);
        sleep (1);          //  Do some 'work'
        zstr_send(responder, "0");
        zstr_free(&str);
    }
    return 0;
}