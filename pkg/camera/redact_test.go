package camera

import (
	"errors"
	"strings"
	"testing"
)

func TestRedactURLsAndSecrets(t *testing.T) {
	t.Setenv(EnvUser, "")
	t.Setenv(EnvPassword, "")
	const pass = "s3cr3t-pa55"
	clean, user, got := Credentials("rtsp://operator:"+pass+"@cam.local:554/stream", "", "")
	if user != "operator" || got != pass {
		t.Fatalf("credentials %q %q", user, got)
	}
	if strings.Contains(clean, "operator") || strings.Contains(clean, pass) {
		t.Fatalf("URL keeps credentials: %s", clean)
	}
	for _, s := range []string{
		"dial rtsp://operator:" + pass + "@cam.local/stream: refused",
		"bad http://x/y?password=" + pass,
		"401 for " + pass,
		"basic b3BlcmF0b3I6czNjcjN0LXBhNTU= rejected",
	} {
		out := Redact(s)
		if strings.Contains(out, pass) || strings.Contains(out, "b3BlcmF0b3I6") {
			t.Errorf("%q -> %q", s, out)
		}
	}
	err := RedactError(errors.New("connect rtsp://operator:" + pass + "@cam.local/ failed"))
	if strings.Contains(err.Error(), pass) {
		t.Errorf("error leaks: %v", err)
	}
	if RedactError(nil) != nil {
		t.Error("RedactError(nil) must be nil")
	}
}

func TestCredentialsFromEnvironment(t *testing.T) {
	t.Setenv(EnvUser, "envuser")
	t.Setenv(EnvPassword, "env-secret")
	clean, u, p := Credentials("rtsp://cam.local/s", "", "")
	if clean != "rtsp://cam.local/s" || u != "envuser" || p != "env-secret" {
		t.Fatalf("got %q %q %q", clean, u, p)
	}
	_, u, p = Credentials("rtsp://a:b@cam.local/s", "", "")
	if u != "a" || p != "b" {
		t.Fatalf("URL must win over the environment: %q %q", u, p)
	}
	if strings.Contains(Redact("failed with env-secret"), "env-secret") {
		t.Error("environment password not redacted")
	}
}
