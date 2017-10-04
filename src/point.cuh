struct PointStruct {
    float3 pos;
    float3 vel;
};

class Point {
    public:
        int size;
        int count;
        GLuint pointShaderProgram;
        GLuint VBO;
        struct cudaGraphicsResource *cuda_vbo_resource;
        PointStruct* data;
        Point(int _count);
        PointStruct *dptr;
        
        void bindVBO();
        void unbindVBO();
        void draw();
        void tick();
};