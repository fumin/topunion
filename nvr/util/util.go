package util

import (
	"fmt"
	"strconv"
	"strings"
	"text/template/parse"
	"time"

	"github.com/pkg/errors"
)

const (
	// VLCUDPLen is the UDP packet length required by the VLC player.
	VLCUDPLen = 1316
)

var (
	ErrNotFound = errors.Errorf("not found")
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
