package main

import (
	"fmt"
	"os"
	"slices"

	"github.com/aep/gaal/display"

	"github.com/aep/moxgo/cmd"
	"github.com/aep/moxgo/pkg/camera"
)

func main() {
	run := func() {
		if err := cmd.Execute(); err != nil {
			fmt.Fprintln(os.Stderr, "Error:", camera.Redact(err.Error()))
			os.Exit(1)
		}
	}
	// A window needs the platform's event loop on the main thread (macOS).
	if slices.Contains(os.Args, "--display") {
		display.Main(run)
		return
	}
	run()
}
