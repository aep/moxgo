package cmd

import "testing"

func TestCameraFlagAndArgs(t *testing.T) {
	reset := func() { runCamera, runCameraID = false, 0 }
	reset()
	f := cameraFlag{}
	if err := f.Set("true"); err != nil || !runCamera {
		t.Fatalf("true: %v %v", err, runCamera)
	}
	reset()
	if err := f.Set("3"); err != nil || !runCamera || runCameraID != 3 {
		t.Fatalf("3: %v %v %d", err, runCamera, runCameraID)
	}
	reset()
	if err := f.Set("false"); err != nil || runCamera {
		t.Fatalf("false: %v %v", err, runCamera)
	}
	if err := f.Set("banana"); err == nil {
		t.Fatal("banana accepted")
	}
	reset()
	runCamera = true
	m, err := parseRunArgs([]string{"yolo11n", "2"})
	if err != nil || m != "yolo11n" || runCameraID != 2 {
		t.Fatalf("%q %v %d", m, err, runCameraID)
	}
	m, err = parseRunArgs([]string{"4", "yolo11n"})
	if err != nil || m != "yolo11n" || runCameraID != 4 {
		t.Fatalf("%q %v %d", m, err, runCameraID)
	}
	reset()
	if err := runArgs(runCmd, []string{"a", "b"}); err == nil {
		t.Fatal("two args without --camera accepted")
	}
}
