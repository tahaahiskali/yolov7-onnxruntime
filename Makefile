CC=g++
CFLAGS=-Wall
SOURCES=main.cpp
EXECUTABLE=yolov7-onnxruntime

LIBS = -lonnxruntime -lopencv_dnn -lopencv_imgproc -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_videoio

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(LIBS) -o $@

clean:
	rm -rf $(EXECUTABLE)
