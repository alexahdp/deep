struct PointStruct {
    float3 pos;
    float3 vel;
};

class Point {
    public:
        int COUNT;
        int SIZE;
        int count;
        GLuint pointShaderProgram;
        GLuint VBO;
        struct cudaGraphicsResource *cuda_vbo_resource;
        PointStruct* data;
        Point(int _count);
        PointStruct *dptr;
        
        void bindVBO();
        void unbindVBO();
        int size();
        void draw();
        void tick();
        void add(float3 pos, float3 vel);
        void d2h();
        void h2d();
};