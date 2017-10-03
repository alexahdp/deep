struct Point {
    float3 pos;
};

class Ppoint {
    public:
        int size;
        int count;
        GLuint pointShaderProgram;
        GLuint VBO;
        struct cudaGraphicsResource *cuda_vbo_resource;
        Point* data;
        Ppoint(int _count);
        Point *dptr;
        
        void bindVBO();
        void unbindVBO();
        void draw();
        void tick();
};