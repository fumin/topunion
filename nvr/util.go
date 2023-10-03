package nvr

import (
	"fmt"
	"io"
	"log"
	"os/exec"
	"time"

	"github.com/pkg/errors"
)

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
		for {
			err := fn(q.req)
			if err != nil {
				log.Printf("%+v", err)
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

func RunProc(quit chan struct{}, duration time.Duration, shutdown func(stdin io.WriteCloser), program string, arg ...string) error {
	cmd := exec.Command(program, arg...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return errors.Wrap(err, "")
	}
	if err := cmd.Start(); err != nil {
		return errors.Wrap(err, "")
	}
	waitC := make(chan error)
	go func() {
		err := cmd.Wait()
		waitC <- err
	}()

	// Let process run for either its duration or early quited.
	select {
	case err := <-waitC:
		return err
	case <-time.After(duration):
	case <-quit:
	}

	// Attempt to cleanly stop process.
	shutdown(stdin)
	// Wait a while for the process to exit.
	const exitWaitSecs = 3
	select {
	case err := <-waitC:
		return err
	case <-time.After(exitWaitSecs * time.Second):
	}

	// Force kill process.
	msg := fmt.Sprintf("unable to exit in %d seconds", exitWaitSecs)
	if err := cmd.Process.Kill(); err != nil {
		msg += fmt.Sprintf(", kill err: %s", err)
	}
	return errors.Errorf(msg)
}
