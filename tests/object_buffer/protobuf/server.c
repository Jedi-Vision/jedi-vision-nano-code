/**
 * Request-and-reply server example code using ZeroMQ.
 */

#include <czmq.h>
#include "../../../src/jv/pb/objectrep.h"
#include "../../../src/c_utils/c_utils.h"

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
       // Receive all frames if multipart message
        size_t len = 0;
        u_int8_t *buffer = NULL;
        zframe_t *frame = zframe_recv(responder);
        zmsg_t *msg = NULL;
        if (frame && zframe_more(frame)) {
            // Multipart message: collect all frames
            msg = zmsg_recv(responder);
            zmsg_prepend(msg, &frame);
            // Concatenate all frames into one buffer
            size_t total_len = 0;
            for (zframe_t *f = zmsg_first(msg); f; f = zmsg_next(msg)) {
                total_len += zframe_size(f);
            }
            u_int8_t *buffer = malloc(total_len);
            if (!buffer) {
                zmsg_destroy(&msg);
                fprintf(stderr, "malloc failed\n");
                break;
            }
            size_t offset = 0;
            for (zframe_t *f = zmsg_first(msg); f; f = zmsg_next(msg)) {
                memcpy(buffer + offset, zframe_data(f), zframe_size(f));
                offset += zframe_size(f);
            }
            zmsg_destroy(&msg);
            len = total_len;
        } else if (frame) {
            // Single frame
            len = zframe_size(frame);
            buffer = malloc(len);
            if (!buffer) {
                zframe_destroy(&frame);
                fprintf(stderr, "malloc failed\n");
                break;
            }
            memcpy(buffer, zframe_data(frame), len);
            zframe_destroy(&frame);
        } else {
            // interrupted or error
            break;
        }
        // Allocate workspace for decoding
        uint8_t workspace[1024];
        struct jv_object_rep_data_t *object = jv_object_rep_data_new(workspace, sizeof(workspace));
        if (object == NULL) {
            fprintf(stderr, "Failed to allocate object representation data.\n");
            continue;
        }

        // Decode the received string into the protobuf object
        printf("%d\n", buffer);
        int decode_result = jv_object_rep_data_decode(object, buffer, len);
        if (decode_result != 0) {
            fprintf(stderr, "Failed to decode received data.\n");
            continue;
        }

        // Print the decoded object for debugging
        printf("Decoded object: id=%d", object->objects.items_p->id);

        // Process the decoded object (if needed)
        // Example: printf("Decoded object successfully.\n");
        sleep (1);          //  Do some 'work'
        zstr_send(responder, "0");
        zstr_free(&buffer);

    }
    return 0;
}