CC=g++
DEST=build
COMPILER_FLAGS=-Wall -Wextra -std=c++11
SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)

main: $(OBJECTS)
	$(CC) \
	$(COMPILER_FLAGS) \
	-o $(DEST)/main $(DEST)/$(OBJECTS) \
	-lc -ldl -lglfw3 -lGLEW -lGLU -lGL -lXinerama -lXcursor -lX11 -lXrandr -lXi -lXxf86vm

main.o: $(SOURCES)
	$(CC) $(COMPILER_FLAGS) -I /usr/include/GL -I /usr/local/include -c $(SOURCES) -o $(DEST)/$(OBJECTS)

clean:
	rm -f build/*


# ======================================
# MAKEFLAGS=-r
# CC=g++
# CFLAGS=-Wall -Wextra -std=c++11
# SRC_DIR=./src
# BIN_DIR=./bin
# BUILD_DIR=./build
# TARGET=main


# abs_srcdir := $(abspath $(SRC_DIR))
# abs_bindir := $(abspath $(BIN_DIR))
# abs_builddir := $(abspath $(BUILD_DIR))

# OBJECTS=$(abs_srcdir/:.cpp=.o)
# MKDIR_P=mkdir -p

# all: $(abs_bindir)/$(TARGET)

# #dir: ${abs_builddir} ${abs_bindir}

# $(abs_bindir)/$(TARGET): $(OBJECTS)
# 	$(CC) $(OBJECTS) $(CFLAGS) -o $@ \
# 	-lc -ldl -lglfw3 -lGLEW -lGLU -lGL -lXinerama -lXcursor -lX11 -lXrandr -lXi -lXxf86vm

# $(OBJECTS): $(abs_builddir)/%.o : $(abs_srcdir)/%.cpp
# 	@$(CC) \
# 	-I /usr/include/GL -I /usr/local/include \
# 	$(CFLAGS) -c $< -o $@

# ${abs_builddir}:
# 	${MKDIR_P} ${abs_builddir}

# ${abs_bindir}:
# 	${MKDIR_P} ${abs_bindir}

# clean:
# 	rm -f build/*