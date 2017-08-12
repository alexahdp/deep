// #define GLEW_STATIC
#include <stdio.h>
#include <vector>
#include <iostream>
#include <glew.h>
#include <GLFW/glfw3.h>

/*
План
 - двигать шарики
 - линии привязаны к шарикам
 - двигать шарики через cuda
 */

// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

// Window dimensions
const GLuint WIDTH = 800, HEIGHT = 600;

GLFWwindow* window;
GLuint lineShaderProgram;
GLuint pointShaderProgram;

std::vector<float> pos;
std::vector<float> vel;

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    std::cout << key << std::endl;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
    pos.push_back(0.4);
    pos.push_back(0.4);
    pos.push_back(0);
}

int init() {
    std::cout << "Starting GLFW context, OpenGL 4.5" << std::endl;
    // Init GLFW
    glfwInit();
    // Set all the required options for GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Create a GLFWwindow object that we can use for GLFW's functions
    window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", nullptr, nullptr);
    if (window == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    // Set the required callback functions
    glfwSetKeyCallback(window, key_callback);

    // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
    glewExperimental = GL_TRUE;
    // Initialize GLEW to setup the OpenGL Function pointers
    
    GLenum glewinit = glewInit();
    if (glewinit != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << glewGetErrorString(glewinit) << std::endl;
        return -1;
    }

    // Define the viewport dimensions
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    
    return 0;
}

GLuint getLineShaderProgram() {
    // Shaders
    const GLchar* vertexShaderSource = "#version 450 core\n"
        "in vec3 vertex_position;\n"
        "void main(void) {\n"
            "gl_Position = vec4(vertex_position, 1.0);\n"
        "}\n\0";

    const GLchar* fragmentShaderSource = "#version 450 core\n"
        "void main(void) {\n"
          "gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "}\n\0";
    
    // Build and compile our shader program
    // Vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // Check for compile time errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // Check for compile time errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Link shaders
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}

GLuint getPointShaderProgram() {
    // Shaders
    const GLchar* vertexShaderSource = "#version 450 core\n"
        "in vec3 vertex_position;\n"
        "void main(void) {\n"
            "gl_Position = vec4(vertex_position, 1.0);\n"
            "gl_PointSize = 5.0;\n"
        "}\n\0";

    const GLchar* fragmentShaderSource = "#version 450 core\n"
        "void main(void) {\n"
        "  vec2 pos = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);\n"
        "  float mag = dot(pos, pos);\n"
          "if (mag > 1.0) discard;\n"
          "gl_FragColor = vec4(1.0, .0, .0, 1.0);\n"
        "}\n\0";
    
    // Build and compile our shader program
    // Vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // Check for compile time errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // Check for compile time errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    // Link shaders
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}

void drawLines() {
    glEnable(GL_LINE_SMOOTH);
    glUseProgram(lineShaderProgram);
    
    glLineWidth(2.0);
    glDrawArrays(GL_LINES, 0, 2);
    
    glDisable(GL_LINE_SMOOTH);
}

void drawPoints() {
    glUseProgram(pointShaderProgram);
    glPointSize(10.0);
    //glVertexPointer(2, GL_FLOAT, 0, pointVertex);
    glDrawArrays(GL_POINTS, 0, pos.size() / 3);
}


// The MAIN function, from here we start the application and run the game loop
int main() {
    
    int res = init();
    if (res != 0) {
        return 1;
    }
    
    lineShaderProgram = getLineShaderProgram();
    pointShaderProgram = getPointShaderProgram();
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, 0, HEIGHT, 0, 1);
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    
    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    GLfloat lineVertices[] = {
        -0.2, 0, 0,
        -0.2, 0.5, 0
    };
    
    pos = {
        0.1, 0.1, 0,
        0.2, 0.2, 0,
        0.8, 0.8, 0
    };
    
    vel = {
        0.01, 0.01, 0,
        0.01, 0.02, 0,
        -0.005, -0.025, 0,
    };
    
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
    
    while (!glfwWindowShouldClose(window)) {
        //std::cout << pos.size() << std::endl;
        
        for (int i = 0; i < pos.size(); i++) {
            pos[i] += vel[i];
        }
        
        glfwPollEvents();
        
        glClearColor(0, 0, 0, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableVertexAttribArray(0);
        
        glBufferData(GL_ARRAY_BUFFER, sizeof(lineVertices), lineVertices, GL_STATIC_DRAW);
        drawLines();
        
        glBufferData(GL_ARRAY_BUFFER, pos.size() * sizeof(float), pos.data(), GL_DYNAMIC_DRAW);
        
        drawPoints();
        
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableVertexAttribArray(0);
        
        // Swap the screen buffers
        glfwSwapBuffers(window);
    }

    // Properly de-allocate all resources once they've outlived their purpose
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    
    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();
    return 0;
}
