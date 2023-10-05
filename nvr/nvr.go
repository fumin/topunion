package nvr

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/pkg/errors"
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

type counter struct {
	sync.Mutex
	stdin io.ReadCloser
}

func (c *counter) setStdin(stdin io.ReadCloser) {
	c.Lock()
	c.stdin = stdin
	c.Unlock()
}

func (c *counter) write(p []byte) (int, error) {
	c.Lock()
	n, err := c.stdin.Write(p)
	c.Unlock()
	return n, err
}

type processOp struct {
	Type     string `json:"type"`
	DstTrack string `json:"dstTrack"`
	Src      string `json:"src"`
}

func (c *counter) process(op processOp) error {
	b, err := json.Marshal(op)
	if err != nil {
		return errors.Wrap(err, "")
	}

	if _, err := c.write(append(b, '\n')); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func Count(quit *Quiter, dst, src string, onStart func(int, time.Time)) {
	cnter := &counter{}
	runCount := func(quit chan struct{}) error {
		program := "python"
		cmd := exec.Command(program, arg...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return errors.Wrap(err, "")
		}
		outerrSize := 1024 * 1024
		cmd.Stdout = NewByteQueue(outerrSize)
		cmd.Stderr = NewByteQueue(outerrSize)
		if err := cmd.Start(); err != nil {
			return errors.Wrap(err, "")
		}
		onStart(cmd.Process.Pid, now)
		cnter.setStdin(stdin)

		shutdown := func() error {
			_, err := cnter.write([]byte(`{"type": "quit"}` + "\n"))
			return err
		}

		runDuration := 999 * 365 * 24 * time.Hour
		cleanupDuration := 3 * time.Seconds
		if err := RunProc(quit, cmd, runDuration, shutdown, cleanupDuration); err != nil {
			outB := cmd.Stdout.(*ByteQueue).Slice()
			errB := cmd.Stderr.(*ByteQueue).Slice()
			return errors.Wrap(err, fmt.Sprintf("stdout: %s, stderr: %s", outB, errB))
		}
		return nil
	}
	countC := make(chan error)
	go func() {
		countC <- quit.Loop(runCount)
	}()

	go func() {
		for {
			ops := getUnprocessed()
			for _, op := range ops {
				if err := cnter.process(op)
			}
		}
	}()

	countErr := <-countC
}

func RecordVideoFn(dir string, getInput func() (string, error), onStart func(int, time.Time)) quiterRunFn {
	const program = "ffmpeg"
	const hlsFlags = "" +
		// Append to the same HLS index file.
		"append_list" +
		// Use microseconds in segment filename.
		"+second_level_segment_duration"
	run := func(quit chan struct{}) error {
		input, err := getInput()
		if err != nil {
			return errors.Wrap(err, "")
		}

		now := time.Now().In(time.UTC)
		endT := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
		endT = endT.AddDate(0, 0, 1)
		runDuration := endT.Sub(now)

		dayDir := filepath.Join(dir, now.Format("20060102"))
		if err := os.MkdirAll(dayDir, os.ModePerm); err != nil {
			return errors.Wrap(err, "")
		}
		segmentFName := filepath.Join(dayDir, "%s_%%06t.ts")
		// strftime does not support "%s" in windows.
		if runtime.GOOS == "windows" {
			segmentFName = filepath.Join(dayDir, "%Y%m%d_%H%M%S_%%06t.ts")
		}
		indexFName := filepath.Join(dayDir, "index.m3u8")
		arg := []string{
			"-i", input,
			// No audio.
			"-an",
			// 10 seconds per segment.
			"-hls_time", "10",
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
		cmd := exec.Command(program, arg...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return errors.Wrap(err, "")
		}
		outerrSize := 1024 * 1024
		cmd.Stdout = NewByteQueue(outerrSize)
		cmd.Stderr = NewByteQueue(outerrSize)
		if err := cmd.Start(); err != nil {
			return errors.Wrap(err, "")
		}
		onStart(cmd.Process.Pid, now)

		// ffmpeg quits by sending q.
		shutdown := func() error {
			_, err := stdin.Write([]byte("q"))
			return err
		}
		// ffmpeg cleans up itself pretty fast.
		const cleanupDuration = 3 * time.Second

		if err := RunProc(quit, cmd, runDuration, shutdown, cleanupDuration); err != nil {
			outB := cmd.Stdout.(*ByteQueue).Slice()
			errB := cmd.Stderr.(*ByteQueue).Slice()
			return errors.Wrap(err, fmt.Sprintf("stdout: %s, stderr: %s", outB, errB))
		}
		return nil
	}
	return run
}
