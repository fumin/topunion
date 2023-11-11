package ffmpeg

import (
	"bytes"
	"context"
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

func FFProbe(ctx context.Context, input []string) (ProbeOutput, error) {
	cmd := exec.CommandContext(ctx, "ffprobe", append([]string{"-print_format", "json", "-show_format", "-show_streams"}, input...)...)
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

// FixFFProbeArg fixes inputs to the ffprobe command.
// It removes options that are valid for ffmpeg, but not for ffprobe.
func FixFFProbeInput(input []string) []string {
	fixed := make([]string, 0, len(input))
	i := 0
	for i < len(input) {
		inp := input[i]

		switch inp {
		case "-stream_loop":
			i += 2
		case "-re":
			i += 1
		default:
			fixed = append(fixed, inp)
			i++
		}
	}
	return fixed
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
