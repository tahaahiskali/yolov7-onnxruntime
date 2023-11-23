CC=g++
CFLAGS=-Wall
SOURCES=main.cpp
EXECUTABLE=yolov7-onnxruntime

LIBS = `pkg-config --libs opencv4` -lonnxruntime 

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(LIBS) -o $@

clean:
	rm -rf $(EXECUTABLE)
