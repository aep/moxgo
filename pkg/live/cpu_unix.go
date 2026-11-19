//go:build unix

package live

import (
	"syscall"
	"time"
)

func processCPU() time.Duration {
	var r syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &r) != nil {
		return 0
	}
	return time.Duration(r.Utime.Nano() + r.Stime.Nano())
}
