package onnx

import (
	"fmt"
	"runtime"
	"strings"
	"unsafe"
)

// Plugin execution providers are shared libraries that plug into a stock
// ONNX Runtime (1.22 and later) instead of being built into it, such as
// Qualcomm's QNN EP (github.com/onnxruntime/onnxruntime-qnn). They are
// registered in the process's OrtEnv, which then lists the devices they
// can run on; a session uses one by appending those devices.

// minPluginAPIVersion is the first ORT API version with plugin EPs.
const minPluginAPIVersion = 22

// RegisterEPLibrary registers the plugin execution provider library at
// path under name, the name an ExecutionProvider then uses to select it.
// Registering a name again is a no-op.
func (rt *Runtime) RegisterEPLibrary(name, path string) error {
	if rt.APIVersion < minPluginAPIVersion {
		return fmt.Errorf("onnx: plugin execution providers need ONNX Runtime 1.%d or later, have %s",
			minPluginAPIVersion, rt.versionStr)
	}
	rt.pluginMu.Lock()
	defer rt.pluginMu.Unlock()
	if _, ok := rt.plugins[name]; ok {
		return nil
	}
	env, err := rt.ensureEnv()
	if err != nil {
		return err
	}
	nameC, pathC := cstring(name), cstring(path)
	err = rt.call(slotRegisterExecutionProviderLibrary, env,
		uintptr(unsafe.Pointer(&nameC[0])), uintptr(unsafe.Pointer(&pathC[0])),
		0, 0, 0, 0, 0)
	runtime.KeepAlive(nameC)
	runtime.KeepAlive(pathC)
	if err != nil {
		return fmt.Errorf("register execution provider %s (%s): %w", name, path, err)
	}
	if rt.plugins == nil {
		rt.plugins = map[string]string{}
	}
	rt.plugins[name] = path
	return nil
}

// PluginEPs returns the registered plugin execution providers, by name.
func (rt *Runtime) PluginEPs() map[string]string {
	rt.pluginMu.Lock()
	defer rt.pluginMu.Unlock()
	out := make(map[string]string, len(rt.plugins))
	for k, v := range rt.plugins {
		out[k] = v
	}
	return out
}

// ensureEnv creates the runtime's OrtEnv. ORT has one OrtEnv per process
// and CreateEnv adds a reference to it, so the sessions' CreateEnv calls
// return this same one; holding it keeps registered plugins alive while
// no session is open. Callers hold pluginMu.
func (rt *Runtime) ensureEnv() (uintptr, error) {
	if rt.env != 0 {
		return rt.env, nil
	}
	logid := cstring("gomax")
	var env uintptr
	err := rt.call(slotCreateEnv, uintptr(OrtLoggingLevelError),
		uintptr(unsafe.Pointer(&logid[0])), uintptr(unsafe.Pointer(&env)),
		0, 0, 0, 0, 0)
	runtime.KeepAlive(logid)
	if err != nil {
		return 0, err
	}
	rt.env = env
	return env, nil
}

func (rt *Runtime) releaseEnv() {
	rt.pluginMu.Lock()
	defer rt.pluginMu.Unlock()
	if rt.env != 0 {
		rt.release(slotReleaseEnv, rt.env)
		rt.env = 0
	}
	rt.plugins = nil
}

// isPluginEP reports whether name selects a registered plugin EP.
func (rt *Runtime) isPluginEP(name string) bool {
	rt.pluginMu.Lock()
	defer rt.pluginMu.Unlock()
	_, ok := rt.plugins[name]
	return ok
}

// pluginDevices returns the OrtEpDevices of the plugin EP name: those whose
// EP name is name, or name with ORT's "ExecutionProvider" suffix.
func (rt *Runtime) pluginDevices(name string) ([]uintptr, error) {
	rt.pluginMu.Lock()
	env := rt.env
	rt.pluginMu.Unlock()
	var devs uintptr
	var n uintptr
	if err := rt.call(slotGetEpDevices, env,
		uintptr(unsafe.Pointer(&devs)), uintptr(unsafe.Pointer(&n)),
		0, 0, 0, 0, 0); err != nil {
		return nil, err
	}
	epName := rt.vtable(slotEpDeviceEpName)
	var out []uintptr
	var seen []string
	for i := uintptr(0); i < n; i++ {
		dev := *(*uintptr)(unsafe.Pointer(devs + i*unsafe.Sizeof(uintptr(0)))) //nolint:govet
		got := goString(ortCall(epName, dev, 0, 0, 0, 0, 0, 0, 0))
		seen = append(seen, got)
		if got == name || got == name+"ExecutionProvider" || strings.TrimSuffix(got, "ExecutionProvider") == name {
			out = append(out, dev)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("execution provider %s has no device here (devices: %s)", name, strings.Join(seen, ", "))
	}
	return out, nil
}

// appendEPPlugin appends the plugin EP ep.Name on all its devices.
func (s *Session) appendEPPlugin(ep ExecutionProvider) error {
	rt := s.rt
	devs, err := rt.pluginDevices(ep.Name)
	if err != nil {
		return err
	}
	keys, vals, keep := cstringArrays(ep.Options)
	var keysPtr, valsPtr unsafe.Pointer
	if len(keys) > 0 {
		keysPtr, valsPtr = unsafe.Pointer(&keys[0]), unsafe.Pointer(&vals[0])
	}
	rt.pluginMu.Lock()
	env := rt.env
	rt.pluginMu.Unlock()
	err = rt.call(slotSessionOptionsAppendExecutionProviderV2,
		s.opts, env, uintptr(unsafe.Pointer(&devs[0])), uintptr(len(devs)),
		uintptr(keysPtr), uintptr(valsPtr), uintptr(len(keys)), 0)
	runtime.KeepAlive(devs)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(vals)
	runtime.KeepAlive(keep)
	return err
}

// Preload loads the shared library at path and keeps it loaded, with its
// symbols global, so that libraries loaded later which open it by its bare
// name get this one: the dynamic linker matches libraries already loaded
// by their name before it searches its paths, which on Android do not
// include an app's own directories.
func Preload(path string) error {
	if _, err := dlOpenGlobal(path); err != nil {
		return fmt.Errorf("preload %s: %w", path, err)
	}
	return nil
}
