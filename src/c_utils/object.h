#ifndef OBJECT_H
#define OBJECT_H

typedef struct ObjectCoordData {
    int id;
    int label;
    float x_2d;   // Normalized [0, 1]
    float y_2d;   // Normalized [0, 1]
    float depth;  // Depth in meters
} ObjectCoordData;

typedef struct ObjectRepData {
    int frame_number;
    float timestamp_ms;
    ObjectCoordData* objects;
} ObjectRepData;

#endif // OBJECT_H