package nvr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"

	"github.com/pkg/errors"
)

type ProbeOutput struct {
	Stdout string
	Stderr string
	Format struct {
		Filename   string `json:"filename"`
		FormatName string `json:"format_name"`
	} `json:"format"`
}

func FFProbe(input string) (ProbeOutput, error) {
	cmd := exec.Command("ffprobe", "-print_format", "json", "-show_format", "-show_streams", input)
	stdout, stderr := bytes.NewBuffer(nil), bytes.NewBuffer(nil)
	cmd.Stdout, cmd.Stderr = stdout, stderr
	err := cmd.Run()
	stdoutB, stderrB := stdout.Bytes(), stderr.Bytes()
	if err != nil {
		return ProbeOutput{}, errors.Wrap(err, fmt.Sprintf("stdout: %s, stderr: %s", stdoutB, stderrB))
	}

	var out ProbeOutput
	if err := json.Unmarshal(stdoutB, &out); err != nil {
		return ProbeOutput{}, errors.Wrap(err, fmt.Sprintf("stdout: %s, stderr: %s", stdoutB, stderrB))
	}
	out.Stdout, out.Stderr = string(stdoutB), string(stderrB)
	return out, nil
}
