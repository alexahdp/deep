struct pos2 {
    float3 f;
    float3 t;
};
struct LineStruct {
    pos2 pos;
};

class Line {
    public:
        int size;
        int count;
        GLuint lineShaderProgram;
        GLuint VBO;
        struct cudaGraphicsResource *cuda_vbo_resource;
        LineStruct* data;
        Line(int _count);
        LineStruct *dptr;
        
        void bindVBO();
        void unbindVBO();
        void draw();
        void tick();
};
