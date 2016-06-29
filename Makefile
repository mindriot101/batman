SOURCES := c_src/_rsky.c c_src/_eclipse.c c_src/_nonlinear_ld.c c_src/light_curve.c
HEADER := c_src/batman.h
OBJECTS := $(SOURCES:.c=.o)
SONAME := libbatman.so
ANAME := libbatman.a
RUN := a.out
PREFIX ?= /usr/local

.PHONY: clean test install

all: $(RUN) $(ANAME)
	
$(RUN): testing/main.c $(SONAME)
	$(CC) $< -L. -lbatman -lm -Ic_src -o $@ -Wall -Wextra

$(SONAME): $(OBJECTS)
	$(CC) -shared -o $@ $^

$(ANAME): $(OBJECTS)
	ar rcs $@ $(OBJECTS)

%.o: %.c
	$(CC) -c -fPIC $< -o $@

install: $(ANAME) $(SONAME) $(HEADER)
	mkdir -p $(PREFIX)/include $(PREFIX)/lib
	install -m 0644 $(ANAME) $(SONAME) $(PREFIX)/lib
	install -m 0644 $(HEADER) $(PREFIX)/include

test: $(RUN)
	time ./$(RUN)
	time python ./testing/test.py

clean:
	@-rm $(RUN) 2>/dev/null || true
	@-rm $(SONAME) 2>/dev/null || true
	@-rm $(ANAME) 2>/dev/null || true
	@-rm $(OBJECTS) 2>/dev/null || true
