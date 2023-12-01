package util

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/pkg/errors"
)

func NewCmd(ctx context.Context, cmdDir string, program string, arg []string) (*exec.Cmd, *os.File, *os.File, string, error) {
	now := time.Now().In(time.UTC)
	id := now.Format(FormatDatetime) + "_" + RandID()
	dir := filepath.Join(cmdDir, now.Format(FormatDate), id)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return nil, nil, nil, "", errors.Wrap(err, "")
	}
	stdout, err := os.Create(filepath.Join(dir, StdoutFilename))
	if err != nil {
		return nil, nil, nil, "", errors.Wrap(err, "")
	}
	stderr, err := os.Create(filepath.Join(dir, StderrFilename))
	if err != nil {
		stdout.Close()
		return nil, nil, nil, "", errors.Wrap(err, "")
	}
	statusPath := filepath.Join(dir, StatusFilename)
	cmd := exec.CommandContext(ctx, program, arg...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	return cmd, stdout, stderr, statusPath, nil
}
