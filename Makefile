MAKEFLAGS=-r
# CC=g++
CC=nvcc
#CFLAGS=-Wall -Wextra -std=c++11
CFLAGS=-std=c++11
SRC_DIR=src
BUILD_DIR=./build
TARGET=bin/main
OBJECTS=$(BUILD_DIR)/%.o
MKDIR_P=mkdir -p

bin/main: $(BUILD_DIR)/main.o $(BUILD_DIR)/lineShader.o $(BUILD_DIR)/pointShader.o $(BUILD_DIR)/point.o
	$(MKDIR_P) bin
	$(CC) \
	$(BUILD_DIR)/point.o $(BUILD_DIR)/main.o $(BUILD_DIR)/lineShader.o $(BUILD_DIR)/pointShader.o \
	$(CFLAGS) -o bin/main \
	-lc -ldl -lglfw3 -lGLEW -lGLU -lGL -lXinerama -lXcursor -lX11 -lXrandr -lXi -lXxf86vm -lglut

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu
	$(MKDIR_P) build
	@$(CC) \
	-I /usr/include/GL -I /usr/local/include -I ./inc \
	$(CFLAGS) -c $< -o $@

$(BUILD_DIR)/point.o: $(SRC_DIR)/point.cu
	$(MKDIR_P) build
	@$(CC) \
	-I /usr/include/GL -I /usr/local/include -I ./inc \
	$(CFLAGS) -c $< -o $@

$(BUILD_DIR)/lineShader.o: $(SRC_DIR)/lineShader.cpp
	$(MKDIR_P) build
	g++ \
	-I /usr/include/GL -I /usr/local/include -I ./inc \
	$(CFLAGS) -c $< -o $@

$(BUILD_DIR)/pointShader.o: $(SRC_DIR)/pointShader.cpp
	$(MKDIR_P) build
	g++ \
	-I /usr/include/GL -I /usr/local/include -I ./inc \
	$(CFLAGS) -c $< -o $@

clean:
	rm -f build/*
	rm -f bin/*



# MAKEFLAGS=-r
# # CC=g++
# CC=nvcc
# #CFLAGS=-Wall -Wextra -std=c++11
# CFLAGS=-std=c++11
# SRC_DIR=src
# #BIN_DIR=./bin
# BUILD_DIR=./build
# TARGET=bin/main
# #TARGET=bin/simpleGL
# OBJECTS=$(BUILD_DIR)/%.o
# MKDIR_P=mkdir -p

# $(TARGET): $(OBJECTS)
# 	$(MKDIR_P) bin
# 	$(CC) $(OBJECTS) $(CFLAGS) -o $@ \
# 	-lc -ldl -lglfw3 -lGLEW -lGLU -lGL -lXinerama -lXcursor -lX11 -lXrandr -lXi -lXxf86vm -lglut

# # $(OBJECTS): $(SRC_DIR)/*.cu
# $(OBJECTS): $(SRC_DIR)/*.cu $(SRC_DIR)/*.cpp
# 	$(MKDIR_P) build
# 	@$(CC) \
# 	-I /usr/include/GL -I /usr/local/include -I ./inc \
# 	$(CFLAGS) -c $< -o $@

# clean:
# 	rm -f build/*
# 	rm -f bin/*