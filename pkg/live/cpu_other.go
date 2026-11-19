//go:build !unix

package live

import "time"

func processCPU() time.Duration { return 0 }
