#ifndef C_UTILS_PARSE_BYTES_H
#define C_UTILS_PARSE_BYTES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double x;
    double y;
    int32_t object_id;
    int label;
} ObjectXYCoordData;

typedef struct {
    ObjectXYCoordData *coords;
    int32_t num_coords;

    // Mask tensor
    char *dtype;   // e.g. "<float32>"
    int ndim;
    int *shape;    // length ndim
    void *data;    // raw pointer to elements (dtype-specific)
    size_t data_bytes;
} ObjectRepData;

/* Parse a buffer containing an object representation.
 * Returns 0 on success, negative on error.
 * buf and out must not be NULL. len is the size of buf in bytes.
 */
int parse_object_rep(const uint8_t *buf, size_t len, ObjectRepData *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* C_UTILS_PARSE_BYTES_H */