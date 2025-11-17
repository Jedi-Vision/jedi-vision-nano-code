/**
 * Request-and-reply server example code using ZeroMQ.
 */

#include <czmq.h>
#include "../../../src/c_utils/c_utils.h"
#include "../../../src/c_utils/parse_bytes.h"


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
        // if (!frame) {
        //     // interrupted or error
        //     break;
        // }
        // len = zframe_size(frame);
        // buffer = malloc(len);
        // if (!buffer) {
        //     zframe_destroy(&frame);
        //     fprintf(stderr, "malloc failed\n");
        //     break;
        // }
        // memcpy(buffer, zframe_data(frame), len);
        // zframe_destroy(&frame);

        ObjectRepData rep;
        if (parse_object_rep(buffer, len, &rep) != 0) {
            fprintf(stderr, "Failed to parse buffer.\n");
            zstr_send(responder, "0");
            zstr_free(&buffer);
            return 1;
        }

        printf("Parsed %d coords:\n", rep.num_coords);
        for (int i = 0; i < rep.num_coords; ++i) {
            ObjectXYCoordData *c = &rep.coords[i];

            // Get the 1D indexing based on 2D shape of data
            int width = rep.shape ? rep.shape[1] : 1;
            int idx = (int)c->y * width + (int)c->x;
            printf("  [%d] x=%f y=%f z=%f id=%d label='%d'\n", i, c->x, c->y, ((float*)rep.data)[idx], c->object_id, c->label);
        }

        if (rep.dtype) {
            printf("Mask: dtype=%s ndim=%d bytes=%zu\n", rep.dtype, rep.ndim, rep.data_bytes);
            if (rep.data && strcmp(rep.dtype, "float32") == 0 && rep.data_bytes >= 4) {
                float v;
                memcpy(&v, rep.data, sizeof(float));
                printf("  first element (float32) = %f\n", v);
            }
        }

        // free
        free(rep.coords);
        free(rep.shape);
        free(rep.data);

        zstr_send(responder, "0");
        zstr_free(&buffer);
    }
    return 0;
}