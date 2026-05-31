BUILD     := $(CURDIR)/build
SHIM      := $(CURDIR)/shim
GO_TAGS   := osusergo,netgo
GO_CMD    := .
VERSION   ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS   := -s -w -X github.com/aep/moxgo/pkg/version.Version=$(VERSION)
ANDROID_API ?= 24

NDK_BIN := $(ANDROID_NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin

MAC_FLAGS := -target aarch64-macos-none -isysroot $(MACOS_SDK) \
             -L$(MACOS_SDK)/usr/lib -F$(MACOS_SDK)/System/Library/Frameworks \
             -Wno-incompatible-sysroot

TARGETS := linux_amd64 linux_arm64 windows_amd64 darwin_arm64 android_arm64

# ── per-target config ─────────────────────────────────────────────

CC_linux_amd64   := zig cc -target x86_64-linux-gnu
CXX_linux_amd64  := zig c++ -target x86_64-linux-gnu
GOOS_linux_amd64   := linux
GOARCH_linux_amd64 := amd64
EXT_linux_amd64    :=

CC_linux_arm64   := zig cc -target aarch64-linux-gnu
CXX_linux_arm64  := zig c++ -target aarch64-linux-gnu
GOOS_linux_arm64   := linux
GOARCH_linux_arm64 := arm64
EXT_linux_arm64    :=

CC_windows_amd64  := zig cc -target x86_64-windows-gnu
CXX_windows_amd64 := zig c++ -target x86_64-windows-gnu
GOOS_windows_amd64   := windows
GOARCH_windows_amd64 := amd64
EXT_windows_amd64    := .exe

CC_darwin_arm64   := $(SHIM)/zig-cc-macos $(MAC_FLAGS)
CXX_darwin_arm64  := $(SHIM)/zig-cxx-macos $(MAC_FLAGS)
GOOS_darwin_arm64   := darwin
GOARCH_darwin_arm64 := arm64
EXT_darwin_arm64    :=

CC_android_arm64  := $(NDK_BIN)/aarch64-linux-android$(ANDROID_API)-clang
CXX_android_arm64 := $(NDK_BIN)/aarch64-linux-android$(ANDROID_API)-clang++
GOOS_android_arm64   := android
GOARCH_android_arm64 := arm64
EXT_android_arm64    :=

# ── cross-compile rules ──────────────────────────────────────────

.PHONY: all cross clean fmt test vet vuln lint bench androidtest mactest \
        $(addprefix bin-,$(TARGETS))

define TARGET_RULES

bin-$(1):
	@mkdir -p $(BUILD)/bin
	CGO_ENABLED=1 GOOS=$(GOOS_$(1)) GOARCH=$(GOARCH_$(1)) \
		CC="$(CC_$(1))" CXX="$(CXX_$(1))" \
		go build -tags $(GO_TAGS) -ldflags '$(LDFLAGS)' \
		-o $(BUILD)/bin/moxgo-$(VERSION)-$(1)$(EXT_$(1)) $(GO_CMD)

endef

$(foreach t,$(TARGETS),$(eval $(call TARGET_RULES,$(t))))

cross: $(addprefix bin-,$(TARGETS))

# ── dev targets ───────────────────────────────────────────────────

all: fmt vet lint vuln test

fmt:
	go fmt ./...

test:
	go test ./...

vet:
	go vet -unsafeptr=false ./...

vuln:
	govulncheck ./...

lint:
	golangci-lint run ./...

bench:
	go test -bench=. -benchmem -count=1 -run=^$$ ./pkg/onnx/

