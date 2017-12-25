
CC = g++
CPPFLAGS = -Wall -O3 -fPIC -std=c++11 -march=native
INCLUDES = -I.
LDFLAGS = -pthread

all: ftrl_train ftrl_predict convert_model

#.cpp.o:
#	$(CC) -c $^ $(INCLUDES) $(CPPFLAGS)
src/ftrl_train.o: src/ftrl_train.cpp src/*.h
	$(CC) -c src/ftrl_train.cpp -o $@ $(INCLUDES) $(CPPFLAGS)

src/ftrl_predict.o: src/ftrl_predict.cpp src/*.h
	$(CC) -c src/ftrl_predict.cpp -o $@ $(INCLUDES) $(CPPFLAGS)

src/stopwatch.o: src/stopwatch.cpp src/stopwatch.h
	$(CC) -c src/stopwatch.cpp -o $@ $(INCLUDES) $(CPPFLAGS)

src/convert_model.o: src/convert_model.cpp
	$(CC) -c src/convert_model.cpp -o $@ $(INCLUDES) $(CPPFLAGS)

ftrl_train: src/ftrl_train.o src/stopwatch.o
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

ftrl_predict: src/ftrl_predict.o src/stopwatch.o
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

convert_model: src/convert_model.o
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -f src/*.o ftrl_train ftrl_predict convert_model
