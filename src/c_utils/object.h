#ifndef OBJECT_H
#define OBJECT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ObjectCoordData {
    int id;
    int label;
    float x;   // in meters
    float y;   // in meters
    float depth;  // Depth in meters
} ObjectCoordData;

typedef struct ObjectRepData {
    int frame_number;
    float timestamp_ms;
    ObjectCoordData* objects;
    int32_t num_coords;
} ObjectRepData;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // OBJECT_H