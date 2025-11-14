/**
 * Request-and-reply server example code using ZeroMQ.
 */

#include <czmq.h>
#include "../../src/jv/pb/objectrep.h"

int main(int argc, char const *argv[])
{
    //  Socket to talk to clients
    zsock_t *responder = zsock_new(ZMQ_REP);
    int rc = zsock_bind(responder, "tcp://*:5555");
    assert(rc == 5555);

    while (1) {
        char *str = zstr_recv(responder);
        // Allocate workspace for decoding
        uint8_t workspace[1024];
        struct jv_object_rep_data_t *object = jv_object_rep_data_new(workspace, sizeof(workspace));
        if (object == NULL) {
            fprintf(stderr, "Failed to allocate object representation data.\n");
            continue;
        }

        // Decode the received string into the protobuf object
        printf("%s", str);
        int decode_result = jv_object_rep_data_decode(object, (const uint8_t *)str, strlen(str));
        if (decode_result != 0) {
            fprintf(stderr, "Failed to decode received data.\n");
            continue;
        }

        // Print the decoded object for debugging
        printf("Decoded object: id=%d", object->object_coordinates.items_p->object_id);

        // Process the decoded object (if needed)
        // Example: printf("Decoded object successfully.\n");
        sleep (1);          //  Do some 'work'
        zstr_send(responder, "0");
        zstr_free(&str);
    }
    return 0;
}