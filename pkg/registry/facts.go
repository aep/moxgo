package registry

import (
	"bufio"
	"bytes"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

// Facts describe a device for runtime rules to match: always "os" and
// "arch" (Go's GOOS and GOARCH, e.g. android and arm64), and what else
// DetectFacts finds. Embedders may add their own or override these.
type Facts map[string]string

// androidProps are the Android system properties DetectFacts reports, by
// fact name.
var androidProps = map[string]string{
	"soc.manufacturer": "ro.soc.manufacturer", // e.g. QTI, Google, Mediatek
	"soc.model":        "ro.soc.model",        // e.g. SM8750, Tensor G4
	"board":            "ro.board.platform",   // e.g. sun, zumapro
	"device":           "ro.product.device",   // e.g. pa1q, tegu
	"model":            "ro.product.model",    // e.g. SM-S931B, Pixel 9a
	"manufacturer":     "ro.product.manufacturer",
	"sdk":              "ro.build.version.sdk",
}

// DetectFacts returns this device's facts: os and arch, and on Android
// its SoC, board, device, model, manufacturer and SDK level (see
// androidProps), on Linux "nvidia" = "true" with an NVIDIA driver loaded.
func DetectFacts() Facts {
	f := Facts{"os": runtime.GOOS, "arch": runtime.GOARCH}
	switch runtime.GOOS {
	case "android":
		props := getprops()
		for fact, prop := range androidProps {
			if v := props[prop]; v != "" {
				f[fact] = v
			}
		}
	case "linux":
		if _, err := os.Stat("/proc/driver/nvidia/version"); err == nil {
			f["nvidia"] = "true"
		}
	}
	return f
}

// getprops reads all Android system properties from getprop's
// "[name]: [value]" lines.
func getprops() map[string]string {
	out, err := exec.Command("getprop").Output()
	if err != nil {
		return nil
	}
	return parseGetprop(out)
}

func parseGetprop(out []byte) map[string]string {
	props := map[string]string{}
	sc := bufio.NewScanner(bytes.NewReader(out))
	for sc.Scan() {
		line := sc.Text()
		k, v, ok := strings.Cut(line, "]: [")
		if !ok || !strings.HasPrefix(k, "[") || !strings.HasSuffix(v, "]") {
			continue
		}
		props[k[1:]] = v[:len(v)-1]
	}
	return props
}
