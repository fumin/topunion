package util

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/pkg/errors"
)

const (
	StdoutFilename = "stdout.txt"
	StderrFilename = "stderr.txt"
	StatusFilename = "status.txt"
)

func NewCmdFn(dir string, fn func(context.Context, *os.File, *os.File) (*exec.Cmd, error)) func(context.Context) {
	run := func(ctx context.Context) {
		statusPath := filepath.Join(dir, StatusFilename)
		status, err := os.OpenFile(statusPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Printf("\"%s\" %+v", statusPath, err)
			return
		}
		defer status.Close()
		stdout, err := os.OpenFile(filepath.Join(dir, StdoutFilename), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			onCmdInit(status, err)
			return
		}
		defer stdout.Close()
		stderr, err := os.OpenFile(filepath.Join(dir, StderrFilename), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			onCmdInit(status, err)
			return
		}
		defer stderr.Close()

		cmd, err := fn(ctx, stdout, stderr)
		if err != nil {
			onCmdInit(status, err)
			return
		}

		if err := cmd.Start(); err != nil {
			onCmdInit(status, err)
			return
		}
		onCmdStart(status, cmd)

		err = cmd.Wait()
		onCmdExit(status, cmd, err)
	}
	return run
}

const (
	CmdEventInit  = "Init"
	CmdEventStart = "Start"
	CmdEventExit  = "Exit"
)

type CmdMsg struct {
	T     time.Time
	Pid   int
	Event string
	Err   string `json:",omitempty"`

	ExitCode *int `json:",omitempty"`
}

func onCmdInit(w io.Writer, initErr error) {
	if err := onCmdInitErr(w, initErr); err != nil {
		log.Printf("%+v", err)
	}
}

func onCmdInitErr(w io.Writer, initErr error) error {
	m := CmdMsg{
		T:     time.Now(),
		Event: CmdEventInit,
	}
	if initErr != nil {
		m.Err = fmt.Sprintf("%+v", initErr)
	}
	b, err := json.Marshal(m)
	if err != nil {
		return errors.Wrap(err, "")
	}
	if _, err := w.Write(append(b, '\n')); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func onCmdStart(w io.Writer, cmd *exec.Cmd) {
	if err := onCmdStartErr(w, cmd); err != nil {
		log.Printf("%+v", err)
	}
}

func onCmdStartErr(w io.Writer, cmd *exec.Cmd) error {
	m := CmdMsg{
		T:     time.Now(),
		Pid:   cmd.Process.Pid,
		Event: CmdEventStart,
	}
	b, err := json.Marshal(m)
	if err != nil {
		return errors.Wrap(err, "")
	}
	if _, err := w.Write(append(b, '\n')); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func onCmdExit(w io.Writer, cmd *exec.Cmd, waitErr error) {
	if err := onCmdExitErr(w, cmd, waitErr); err != nil {
		log.Printf("%+v", err)
	}
}

func onCmdExitErr(w io.Writer, cmd *exec.Cmd, waitErr error) error {
	m := CmdMsg{
		T:     time.Now(),
		Pid:   cmd.Process.Pid,
		Event: CmdEventExit,
	}
	code := cmd.ProcessState.ExitCode()
	m.ExitCode = &code
	if waitErr != nil {
		m.Err = fmt.Sprintf("%+v", waitErr)
	}
	b, err := json.Marshal(m)
	if err != nil {
		return errors.Wrap(err, "")
	}
	if _, err := w.Write(append(b, '\n')); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func ReadCmdMsg(fpath string) ([]CmdMsg, error) {
	f, err := os.Open(fpath)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	defer f.Close()

	msgs := make([]CmdMsg, 0)
	dec := json.NewDecoder(f)
	for {
		var m CmdMsg
		err := dec.Decode(&m)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		msgs = append(msgs, m)
	}
	return msgs, nil
}

type ErrorLogger struct {
	fpath string
	f     *os.File
}

func NewErrorLogger(fpath string) *ErrorLogger {
	el := &ErrorLogger{fpath: fpath}
	return el
}

func (el *ErrorLogger) Close() {
	el.closeFile()
}

func (el *ErrorLogger) E(fn func() error) {
	funcErr := fn()
	if funcErr == nil {
		return
	}

	writeErr := func() error {
		msg := struct {
			Time time.Time
			Err  string
		}{}
		msg.Time = time.Now()
		msg.Err = fmt.Sprintf("%+v", funcErr)
		b, err := json.Marshal(msg)
		if err != nil {
			return errors.Wrap(err, "")
		}

		f, err := el.file()
		if err != nil {
			return errors.Wrap(err, "")
		}
		if _, err := f.Write(append(b, '\n')); err != nil {
			return errors.Wrap(err, "")
		}
		if err := f.Sync(); err != nil {
			return errors.Wrap(err, "")
		}
		return nil
	}()
	if writeErr != nil {
		el.closeFile()
		log.Printf("funcErr: %+v, writeErr: %+v", funcErr, writeErr)
	}
}

func (el *ErrorLogger) file() (*os.File, error) {
	if el.f != nil {
		return el.f, nil
	}
	var err error
	el.f, err = os.OpenFile(el.fpath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	return el.f, nil
}

func (el *ErrorLogger) closeFile() {
	if el.f == nil {
		return
	}
	if err := el.f.Close(); err != nil {
		log.Printf("%+v", err)
	}
	el.f = nil
}
