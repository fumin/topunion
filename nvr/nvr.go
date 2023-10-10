package nvr

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/pkg/errors"
)

const (
	IndexM3U8 = "index.m3u8"
	HLSTime   = 10
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
	Src        string
	Device     string
	Mask       struct {
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
		PrevCount int
	}
}

func CountFn(script string, cfg CountConfig, onStart func(int)) loopFn {
	c, err := json.Marshal(cfg)
	if err != nil {
		panic(err)
	}
	arg := []string{script, "-c=" + string(c)}
	const program = "python3"
	run := func(ctx context.Context) error {
		cmd := exec.CommandContext(ctx, program, arg...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return errors.Wrap(err, "")
		}
		outerrSize := 1024 * 1024
		cmd.Stdout = NewByteQueue(outerrSize)
		stderr := NewByteQueue(outerrSize)
		stderr.debug = true
		cmd.Stderr = stderr
		cmd.Cancel = func() error {
			_, err := stdin.Write([]byte("q\n"))
			return err
		}
		cmd.WaitDelay = 3 * 60 * time.Second

		if err := cmd.Start(); err != nil {
			return errors.Wrap(err, "")
		}
		onStart(cmd.Process.Pid)

		err = cmd.Wait()
		if err == context.Canceled {
			return nil
		}
		if err != nil {
			outB := cmd.Stdout.(*ByteQueue).Slice()
			errB := cmd.Stderr.(*ByteQueue).Slice()
			return errors.Wrap(err, fmt.Sprintf("stdout: %s, stderr: %s, program: %s, arg: %#v", outB, errB, program, arg))
		}
		return nil
	}
	return run
}

func RecordVideoFn(dir string, getInput func() (string, error), onStart func(int)) loopFn {
	const program = "ffmpeg"
	const hlsFlags = "" +
		// Append to the same HLS index file.
		"append_list" +
		// Use microseconds in segment filename.
		"+second_level_segment_duration" +
		// Add program start time info.
		"+program_date_time"
	run := func(ctx context.Context) error {
		input, err := getInput()
		if err != nil {
			return errors.Wrap(err, "")
		}

		segmentFName := filepath.Join(dir, "%s_%%06t.ts")
		// strftime does not support "%s" in windows.
		if runtime.GOOS == "windows" {
			segmentFName = filepath.Join(dir, "%Y%m%d_%H%M%S_%%06t.ts")
		}
		indexFName := filepath.Join(dir, IndexM3U8)
		arg := []string{
			"-i", input,
			// No audio.
			"-an",
			// 10 seconds per segment.
			"-hls_time", strconv.Itoa(HLSTime),
			// No limit on number of segments.
			"-hls_list_size", "0",
			// Use strftime syntax for segment filenames.
			"-strftime", "1",
			"-hls_flags", hlsFlags,
			"-hls_segment_filename", segmentFName,
			indexFName,
		}
		// If input is a local file, loop it forever.
		if _, err := os.Stat(input); err == nil {
			arg = append([]string{"-stream_loop", "-1"}, arg...)
		}

		program := "ffmpeg"
		cmd := exec.CommandContext(ctx, program, arg...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return errors.Wrap(err, "")
		}
		outerrSize := 1024 * 1024
		cmd.Stdout = NewByteQueue(outerrSize)
		cmd.Stderr = NewByteQueue(outerrSize)
		cmd.Cancel = func() error {
			_, err := stdin.Write([]byte("q"))
			return err
		}
		cmd.WaitDelay = 10 * time.Second

		if err := cmd.Start(); err != nil {
			return errors.Wrap(err, "")
		}
		onStart(cmd.Process.Pid)

		err = cmd.Wait()
		if err == context.Canceled {
			return nil
		}
		if err != nil {
			outB := cmd.Stdout.(*ByteQueue).Slice()
			errB := cmd.Stderr.(*ByteQueue).Slice()
			return errors.Wrap(err, fmt.Sprintf("stdout: %s, stderr: %s", outB, errB))
		}
		return nil
	}
	return run
}
