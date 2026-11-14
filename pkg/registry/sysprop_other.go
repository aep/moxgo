//go:build !android

package registry

// systemProperty is an Android system property; there are none elsewhere.
func systemProperty(string) string { return "" }
