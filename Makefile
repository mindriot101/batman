SOURCES := c_src/_rsky.c c_src/_eclipse.c c_src/_nonlinear_ld.c
OBJECTS := $(SOURCES:.c=.o)
SONAME := libbatman.so


all: $(SONAME)

$(SONAME): $(OBJECTS)
	$(CC) -shared -o $@ $^

%.o: %.c
	$(CC) -c -fPIC $< -o $@

.PHONY: clean
clean:
	@-rm $(SONAME) 2>/dev/null || true
	@-rm $(OBJECTS) 2>/dev/null || true
