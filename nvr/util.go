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
	size int
	q *deque.Deque[byte]
	b []byte
}

func NewByteQueue(size int) *ByteQueue {
	const traySize = 1024
	q := &ByteQueue{size: size}
	q.q = deque.Make[byte](size + traySize)
	q.b = make([]byte, traySize)
	return q
}

func (q *ByteQueue) ReadFrom(r io.Reader) (int, error) {
	n, err := r.Read(q.b)
	if err != nil {
		return n, errors.Wrap(err, "")
	}
	q.q.PushBackSlice(q.b[:n])

	for q.q.Len() > q.size {
		q.q.RemoveFront()
	}
	return n, nil
}

func (q *ByteQueue) ReadTillEOF(r io.Reader) {
	for {
		_, err := q.ReadFrom(r)
		if err == io.EOF {
			return
		}
	}
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
				// Only print errors every interval to prevent constantly failing functions overwhelming our logs.
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
	wait error
	shutdown error
	cleanup error
	kill error
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

func RunProc(quit chan struct{}, onStart func(int), duration time.Duration, shutdown func(stdin io.WriteCloser) error, program string, arg ...string) ([]byte, []byte, error) {
	stdinR, stdinW, err := os.Pipe()
	if err != nil {
		return nil, nil, errors.Wrap(err, "")
	}
	defer stdinR.Close()
	defer stdinW.Close()
	childFiles := make([]*os.File, 0, 3)
	pa := &os.ProcAttr{}
	pa.Files = childFiles
	pa.Env, _ = os.Environ()
	argv := append([]string{program}, arg...)
	proc, err := os.StartProcess(program, argv, pa)
	if err != nil {
		return nil, nil, errors.Wrap(err, "")
	}

	// Collect stdout and stderr in the background.
	kill := make(chan struct{})
	defer close(kill)
	const stdouterrSize = 4*1024
	stdoutC := make(chan []byte)
	go func() {
		q := NewByteQueue(stdouterrSize)
		q.ReadTillEOF(stdout)
		select {
		case <-kill:
		case stdoutC <- q.Slice():
		}
	}()
	stderrC := make(chan []byte)
	go func() {
		q := NewByteQueue(stdouterrSize)
		q.ReadTillEOF(stderr)
		select {
		case <-kill:
		case stderrC <- q.Slice():
		}
	}()

	if err := cmd.Start(); err != nil {
		return nil, nil, errors.Wrap(err, "")
	}
	onStart(cmd.Process.Pid)

	type outErrWait struct {
		stdout []byte
		stderr []byte
		wait error
	}
	waitC := make(chan outErrWait)
	go func() {
		var oew outErrWait
		oew.stdout = <-stdoutC
		oew.stderr = <-stderrC
		oew.wait = cmd.Wait()
		select {
		case <-kill:
		case waitC <- oew:
		}
	}()

	var runErr runError
	// Let process run for either its duration or early quited.
	select {
	case oew := <-waitC:
		runErr.wait = oew.wait
		return oew.stdout, oew.stderr, runErr.orNil()
	case <-time.After(duration):
	case <-quit:
	}

	// Attempt to cleanly stop process.
	runErr.shutdown = shutdown(stdin)
	// Wait a while for the process to exit.
	const exitWaitSecs = 3
	select {
	case oew := <-waitC:
		runErr.wait = oew.wait
		return oew.stdout, oew.stderr, runErr.orNil()
	case <-time.After(exitWaitSecs * time.Second):
		runErr.cleanup = errors.Errorf("unable to exit in %d seconds", exitWaitSecs)
	}

	// Force kill process.
	runErr.kill = cmd.Process.Kill()
	return nil, nil, runErr.orNil()
}
