#include <stdint.h>
#include <stdlib.h>

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
