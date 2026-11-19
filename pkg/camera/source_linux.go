//go:build linux && (amd64 || arm64)

package camera

func openPlatformSource(index int, cfg SourceConfig) (VideoSource, error) {
	return openV4L2Source(index, cfg)
}
