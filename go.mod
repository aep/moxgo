module github.com/aep/moxgo

go 1.26.3

require (
	connectrpc.com/connect v1.20.0
	github.com/ebitengine/purego v0.11.3
	github.com/schollz/progressbar/v3 v3.19.0
	github.com/spf13/cobra v1.10.2
	go.opentelemetry.io/otel v1.46.0
	go.opentelemetry.io/otel/trace v1.46.0
	golang.org/x/image v0.41.0
	google.golang.org/protobuf v1.36.11
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/aep/gaal v0.0.0
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/go-logr/logr v1.4.4 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/spf13/pflag v1.0.9
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/otel/metric v1.46.0 // indirect
	golang.org/x/sys v0.45.0 // indirect
	golang.org/x/term v0.43.0 // indirect
)

replace github.com/aep/gaal => ../../gorm
