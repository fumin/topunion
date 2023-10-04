package nvr

import (
	"fmt"
	"io"
	"log"
	"os/exec"
	"time"

	"github.com/carlmjohnson/deque"
	"github.com/pkg/errors"
)

type ByteQueue struct {
	q *deque.Deque[byte]
}

func NewByteQueue(size int) *ByteQueue {
	q := &ByteQueue{}
	q.q = deque.Make[byte](size)
	return q
}

func (q *ByteQueue) Write(p []byte) (int, error) {
	q.q.PushBackSlice(p)

	for q.q.Len() > q.size {
		q.q.RemoveFront()
	}
	return len(p), nil
}

func (q *ByteQueue) Slice() []byte {
	return q.q.Slice()
}

type Quiter struct {
	req  chan struct{}
	resp chan error
}

func NewQuiter() *Quiter {
	q := &Quiter{}
	q.req = make(chan struct{})
	q.resp = make(chan error)
	return q
}

func (q *Quiter) Loop(fn func(chan struct{}) error) {
	go func() {
		lastLogT := time.Now().AddDate(-1, 0, 0)
		for {
			err := fn(q.req)
			if err != nil {
				// Only print errors intermitently to prevent constantly failing functions from overwhelming our logs.
				now := time.Now()
				if now.Sub(lastLogT) > time.Minute {
					lastLogT = now
					log.Printf("%+v", err)
				}
			}

			select {
			case <-q.req:
				q.resp <- err
			default:
			}
		}
	}()
}

func (q *Quiter) Quit() error {
	close(q.req)
	return <-q.resp
}

type runError struct {
	wait     error
	shutdown error
	cleanup  error
	kill     error
}

func (err runError) Error() string {
	waitMsg := "<nil>"
	if err.wait != nil {
		waitMsg = err.wait.Error()
	}
	shutdownMsg := "<nil>"
	if err.shutdown != nil {
		shutdownMsg = err.shutdown.Error()
	}
	cleanupMsg := "<nil>"
	if err.cleanup != nil {
		cleanupMsg = err.cleanup.Error()
	}
	killMsg := "<nil>"
	if err.kill != nil {
		killMsg = err.kill.Error()
	}
	return fmt.Sprintf("wait: %s, shutdown: %s, cleanup: %s, kill: %s", waitMsg, shutdownMsg, cleanupMsg, killMsg)
}

func (err runError) orNil() error {
	if err.wait != nil {
		return err
	}
	if err.shutdown != nil {
		return err
	}
	if err.cleanup != nil {
		return err
	}
	if err.kill != nil {
		return err
	}
	return nil
}

func RunProc(quit chan struct{}, duration time.Duration, shutdown func() error, startedCmd *exec.Cmd) error {
	exited := make(chan struct{})
	exitC := make(chan runError)
	go func() {
		var xt runError
		defer func() { exitC <- xt }()

		select {
		case <-exited:
			return
		case <-quit:
		case <-time.After(duration):
		}
		xt.shutdown = shutdown()

		// Wait a while for the process to exit.
		const exitWaitSecs = 3
		select {
		case <-exited:
			return
		case <-time.After(exitWaitSecs * time.Second):
		}
		xt.cleanup = errors.Errorf("unable to exit in %d seconds", exitWaitSecs)

		// Force kill.
		xt.kill = cmd.Process.Kill()
	}()

	var err runError
	err.wait = cmd.Wait()
	close(exited)

	xt := <-exitC
	err.shutdown = xt.shutdown
	err.cleanup = xt.cleanup
	err.kill = xt.kill

	return err.orNil()
}
