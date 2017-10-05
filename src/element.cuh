
class Element {
    public:
        int COUNT;
        int SIZE;
        int count;
        // GLuint pointShaderProgram;
        GLuint VBO;
        struct cudaGraphicsResource *cuda_vbo_resource;
        // PointStruct* data;
        // Point(int _count);
        void *dptr;
        
        Element();
        void bindVBO();
        void unbindVBO();
        // void d2h();
        // void h2d();
        
        //virtual int size();
        //void draw();
        //void tick();
        //void add(float3 pos, float3 vel);
        //virtual void * getdata();
};