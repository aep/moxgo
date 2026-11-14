package registry

import (
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Android apps read system properties through libc's
// __system_property_get: running getprop, as a shell can, does not work
// from an app.

var (
	propGetOnce sync.Once
	propGet     uintptr
)

// propValueMax is PROP_VALUE_MAX, the largest property value with its NUL.
const propValueMax = 92

// systemProperty is Android system property name, "" if unset or
// unreadable.
func systemProperty(name string) string {
	propGetOnce.Do(func() {
		libc, err := purego.Dlopen("libc.so", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			return
		}
		propGet, _ = purego.Dlsym(libc, "__system_property_get")
	})
	if propGet == 0 {
		return ""
	}
	key := append([]byte(name), 0)
	var val [propValueMax]byte
	n, _, _ := purego.SyscallN(propGet, uintptr(unsafe.Pointer(&key[0])), uintptr(unsafe.Pointer(&val[0])))
	if int(n) <= 0 || int(n) > len(val) {
		return ""
	}
	return string(val[:n])
}
