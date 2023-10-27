package nvr

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"text/template/parse"
	"time"

	"github.com/pkg/errors"
)

const ymdhmsFormat = "20060102_150405"

func TimeFormat(inT time.Time) string {
	t := inT.In(time.UTC)
	microSec := t.Nanosecond() / 1e3
	return t.Format(ymdhmsFormat) + "_" + fmt.Sprintf("%06d", microSec)
}

func TimeParse(name string) (time.Time, error) {
	ss := strings.Split(name, "_")
	if len(ss) != 3 {
		return time.Time{}, errors.Errorf("%d", len(ss))
	}
	t, err := time.Parse(ymdhmsFormat, ss[0]+"_"+ss[1])
	if err != nil {
		return time.Time{}, errors.Wrap(err, "")
	}
	microSec, err := strconv.Atoi(ss[2])
	if err != nil {
		return time.Time{}, errors.Wrap(err, "")
	}
	nanoSec := microSec * 1e3

	parsed := time.Date(t.Year(), t.Month(), t.Day(), t.Hour(), t.Minute(), t.Second(), nanoSec, t.Location())
	return parsed, nil
}

func TmplFields(t *parse.Tree) map[string]struct{} {
	fields := make(map[string]struct{})
	for _, n := range t.Root.Nodes {
		action, ok := n.(*parse.ActionNode)
		if !ok {
			continue
		}
		cmds := action.Pipe.Cmds
		if len(cmds) == 0 {
			continue
		}
		args := cmds[0].Args
		if len(args) == 0 {
			continue
		}
		field, ok := args[0].(*parse.FieldNode)
		if !ok {
			continue
		}
		if len(field.Ident) == 0 {
			continue
		}

		fields[field.Ident[0]] = struct{}{}
	}
	return fields
}

func newCmdFn(w io.Writer, fn func(context.Context) (*exec.Cmd, error)) func(context.Context) {
	run := func(ctx context.Context) {
		cmd, err := fn(ctx)
		if err != nil {
			onCmdInit(w, err)
			return
		}

		if err := cmd.Start(); err != nil {
			onCmdInit(w, err)
			return
		}
		onCmdStart(w, cmd)

		err = cmd.Wait()
		onCmdExit(w, cmd, err)
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
