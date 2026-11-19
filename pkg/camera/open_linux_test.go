//go:build linux && (amd64 || arm64)

package camera

import (
	"strings"
	"testing"
)

func TestOpenMissingDevice(t *testing.T) {
	_, err := Open(9999, Options{})
	if err == nil || !strings.Contains(err.Error(), "no such file or directory") {
		t.Fatalf("got %v, want ENOENT", err)
	}
}
