DEVKITPRO ?= /opt/devkitpro
WIIU_CMAKE ?= $(DEVKITPRO)/portlibs/wiiu/bin/powerpc-eabi-cmake
BUILD_DIR ?= build-wiiu

.PHONY: all configure build clean

all: build

configure:
	@$(WIIU_CMAKE) -S . -B $(BUILD_DIR) \
		-DCMAKE_TOOLCHAIN_FILE=$(DEVKITPRO)/cmake/WiiU.cmake \
		-DCMAKE_BUILD_TYPE=Release

build: configure
	@$(WIIU_CMAKE) --build $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)
