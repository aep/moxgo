// Package live runs a camera-to-display pipeline with detection on the side:
// video flows at camera rate and is never held up by inference, which works
// on the newest frame whenever it is free and whose latest results are drawn
// into the video.
package live

import "sync"

// Mailbox holds at most one value: Put replaces what is there, so a slow
// taker always gets the newest. It is safe for concurrent use.
type Mailbox[T any] struct {
	mu     sync.Mutex
	val    T
	full   bool
	closed bool
	ready  chan struct{}
}

// NewMailbox returns an empty mailbox.
func NewMailbox[T any]() *Mailbox[T] {
	return &Mailbox[T]{ready: make(chan struct{}, 1)}
}

// Put stores v. It returns the value it replaced, if any, and never blocks.
// After Close, v itself comes back as replaced.
func (m *Mailbox[T]) Put(v T) (old T, replaced bool) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return v, true
	}
	old, replaced = m.val, m.full
	m.val, m.full = v, true
	m.mu.Unlock()
	select {
	case m.ready <- struct{}{}:
	default:
	}
	return old, replaced
}

// Take waits for a value and removes it. ok is false once closed and empty.
func (m *Mailbox[T]) Take() (v T, ok bool) {
	for {
		m.mu.Lock()
		if m.full {
			v, m.val, m.full = m.val, *new(T), false
			m.mu.Unlock()
			return v, true
		}
		closed := m.closed
		m.mu.Unlock()
		if closed {
			return v, false
		}
		<-m.ready
	}
}

// TryTake is Take that does not wait.
func (m *Mailbox[T]) TryTake() (v T, ok bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.full {
		v, m.val, m.full = m.val, *new(T), false
		return v, true
	}
	return v, false
}

// Close wakes Take for good. It returns a value left inside, if any.
func (m *Mailbox[T]) Close() (v T, ok bool) {
	m.mu.Lock()
	v, ok = m.val, m.full
	m.val, m.full, m.closed = *new(T), false, true
	m.mu.Unlock()
	select {
	case m.ready <- struct{}{}:
	default:
	}
	return v, ok
}
