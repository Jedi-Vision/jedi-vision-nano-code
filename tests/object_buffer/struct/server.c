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
        zframe_t *frame = zframe_recv(responder);
        if (!frame) {
            // interrupted or error
            break;
        }
        size_t len = zframe_size(frame);
        u_int8_t *buffer = malloc(len);
        if (!buffer) {
            zframe_destroy(&frame);
            fprintf(stderr, "malloc failed\n");
            break;
        }
        memcpy(buffer, zframe_data(frame), len);
        zframe_destroy(&frame);

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
            printf("  [%d] x=%f y=%f id=%d label='%d'\n", i, c->x, c->y, c->object_id, c->label);
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