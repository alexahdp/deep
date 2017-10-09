#ifndef LINE_CUH
#define LINE_CUH

#include "point.cuh"

struct pos2 {
    float3 f;
    float3 t;
};
struct LineStruct {
    pos2 pos;
};

struct LineFTStruct {
    int f;
    int t;
};

class Line {
    public:
        int size;
        int count;
        GLuint lineShaderProgram;
        GLuint VBO;
        struct cudaGraphicsResource *cuda_vbo_resource;
        LineStruct* data;
        LineFTStruct* ft;
        Line(int _count);
        LineStruct *dptr;
        LineFTStruct *ftdptr;
        
        void bindVBO();
        void unbindVBO();
        void draw();
        void tick(Point* p1);
};

#endif