package nvr

import (
	"context"
	_ "embed"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/pkg/errors"
)

const (
	IndexM3U8     = "index.m3u8"
	ValueFilename = "v.json"
	HLSTime       = 10

	Python = "python3"

	StdoutFilename = "stdout.txt"
	StderrFilename = "stderr.txt"
	StatusFilename = "status.txt"
)

//go:embed count.py
var countPY string

type Scripts struct {
	Count string
}

func NewScripts(dir string) (Scripts, error) {
	s := Scripts{
		Count: filepath.Join(dir, "count.py"),
	}

	if err := os.MkdirAll(dir, 0775); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}

	if err := os.WriteFile(s.Count, []byte(countPY), 0644); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}

	return s, nil
}

type CountConfig struct {
	TrackIndex string
	TrackLog   string
	Src        string
	AI         struct {
		Smart  bool
		Device string
		Mask   struct {
			Enable bool
			Crop   struct {
				X int
				Y int
				W int
			}
			Mask struct {
				Slope float64
				Y     int
				H     int
			}
		}
		Yolo struct {
			Weights string
			Size    int
		}
		Track struct {
		}
	}
}

func CountFn(dir, script string, cfg CountConfig) func(context.Context) {
	run := func(ctx context.Context, stdout, stderr *os.File) (*exec.Cmd, error) {
		c, err := json.Marshal(cfg)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		arg := []string{script, "-c=" + string(c)}

		cmd := exec.CommandContext(ctx, Python, arg...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		cmd.Stdout = stdout
		cmd.Stderr = stderr
		cmd.Cancel = func() error {
			_, err := stdin.Write([]byte("q\n"))
			return err
		}
		cmd.WaitDelay = 60 * time.Second

		return cmd, nil
	}
	return newCmdFn(dir, run)
}

func RecordVideoFn(dir string, getInput func() ([]string, error)) func(context.Context) {
	const program = "ffmpeg"
	const hlsFlags = "" +
		// Append to the same HLS index file.
		"append_list" +
		// Add segment index in filename to deduplicate filenames within the same second.
		"+second_level_segment_index" +
		// Add program time info explicitly.
		// If we do not, with append_list turned on, ffmpeg will nonetheless insist on adding garbage program time, resulting in a worse situation.
		"+program_date_time"
	run := func(ctx context.Context, stdout, stderr *os.File) (*exec.Cmd, error) {
		input, err := getInput()
		if err != nil {
			return nil, errors.Wrap(err, "")
		}

		segmentFName := filepath.Join(dir, "%s_%%06d.ts")
		// strftime does not support "%s" in windows.
		if runtime.GOOS == "windows" {
			segmentFName = filepath.Join(dir, "%Y%m%d_%H%M%S_%%06d.ts")
		}
		indexFName := filepath.Join(dir, IndexM3U8)
		arg := append(input, []string{
			// No audio.
			"-an",
			// Variable frame rate, otherwise the HLS codec fails.
			"-vsync", "vfr",
			// H264 high profile level 4.2 for maximum support across devices.
			"-profile:v", "high", "-level:v", "4.2",
			// 10 seconds per segment.
			"-hls_time", strconv.Itoa(HLSTime),
			// No limit on number of segments.
			"-hls_list_size", "0",
			// Use strftime syntax for segment filenames.
			"-strftime", "1",
			"-hls_flags", hlsFlags,
			"-hls_segment_filename", segmentFName,
			indexFName,
		}...)

		cmd := exec.CommandContext(ctx, program, arg...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		cmd.Stdout = stdout
		cmd.Stderr = stderr
		cmd.Cancel = func() error {
			_, err := stdin.Write([]byte("q"))
			return err
		}
		cmd.WaitDelay = 10 * time.Second

		return cmd, nil
	}
	return newCmdFn(dir, run)
}
