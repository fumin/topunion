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

type Stream struct {
	Index         int    `json:"index"`
	CodecType     string `json:"codec_type"`
	CodecName     string `json:"codec_name"`
	CodecLongName string `json:"codec_long_name"`
	Width         int    `json:"width"`
	Height        int    `json:"height"`
}

type ProbeOutput struct {
	Stdout  string
	Stderr  string
	Streams []Stream `json:"streams"`
	Format  struct {
		Filename   string `json:"filename"`
		FormatName string `json:"format_name"`
		Duration   float64

		DurationStr string `json:"duration"`
	} `json:"format"`
}

func FFProbe(ctx context.Context, input []string) (ProbeOutput, error) {
	progArg := append([]string{"ffprobe", "-print_format", "json", "-show_format", "-show_streams"}, input...)
	cmd := exec.CommandContext(ctx, progArg[0], progArg[1:]...)
	stdout, stderr := bytes.NewBuffer(nil), bytes.NewBuffer(nil)
	cmd.Stdout, cmd.Stderr = stdout, stderr
	err := cmd.Run()
	stdoutB, stderrB := stdout.Bytes(), stderr.Bytes()
	if err != nil {
		return ProbeOutput{}, errors.Wrap(err, fmt.Sprintf("%s, stdout: %s, stderr: %s", strings.Join(progArg, " "), stdoutB, stderrB))
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
func Escape(options []string) []string {
	escaped := make([]string, 0, len(options))
	for _, s := range options {
		s = strings.ReplaceAll(s, `\`, `\\`)
		escaped = append(escaped, s)
	}
	return escaped
}

// TeeEscape escapes strings inside a tee muxer option.
// https://ffmpeg.org/ffmpeg-formats.html#Options-17
func TeeEscape(options []string) string {
	escaped := make([]string, 0, len(options))
	for _, s := range options {
		s = strings.ReplaceAll(s, `\`, `\\`)
		s = strings.ReplaceAll(s, ":", `\:`)
		escaped = append(escaped, s)
	}
	joined := strings.Join(escaped, ":")
	return joined
}
