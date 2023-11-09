package ffmpeg

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

type ProbeOutput struct {
	Stdout string
	Stderr string
	Format struct {
		Filename   string `json:"filename"`
		FormatName string `json:"format_name"`
		Duration   float64

		DurationStr string `json:"duration"`
	} `json:"format"`
}

func FFProbe(input []string) (ProbeOutput, error) {
	cmd := exec.Command("ffprobe", append([]string{"-print_format", "json", "-show_format", "-show_streams"}, input...)...)
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
	// Live streams may not have duration, so ignore errors.
	out.Format.Duration, _ = strconv.ParseFloat(out.Format.DurationStr, 64)
	return out, nil
}

// Escape escapes ffmpeg strings.
// https://ffmpeg.org/ffmpeg-utils.html#quoting_005fand_005fescaping
func Escape(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	return s
}

// TeeEscape escapes strings inside a tee muxer option.
// https://ffmpeg.org/ffmpeg-formats.html#Options-17
func TeeEscape(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, ":", `\:`)
	return s
}
