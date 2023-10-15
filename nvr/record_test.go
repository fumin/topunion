package nvr

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSameIndex(t *testing.T) {
	t.Parallel()
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer os.RemoveAll(dir)
	scripts, err := NewScripts(filepath.Join(dir, "script"))
	if err != nil {
		t.Fatalf("%+v", err)
	}
	testVid := filepath.Join(dir, "test.mp4")
	cmd := strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 "+testVid, " ")
	if b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		t.Fatalf("%+v %s", err, b)
	}

	// Run src.
	srcDir := filepath.Join(dir, "src")
	if err := os.MkdirAll(srcDir, os.ModePerm); err != nil {
		t.Fatalf("%+v", err)
	}
	getInput := func() (string, error) {
		return testVid, nil
	}
	srcCtx, srcCancel := context.WithCancel(context.Background())
	defer srcCancel()
	go func() {
		RecordVideoFn(srcDir, getInput, io.Discard, io.Discard, io.Discard)(srcCtx)
	}()
	<-time.After(500 * time.Millisecond)
	srcCancel()

	// Run dst.
	c := Count{Src: filepath.Base(srcDir)}
	c.Config.AI.Device = "cpu"
	c = c.Fill(dir)
	if err := c.Prepare(); err != nil {
		t.Fatalf("%+v", err)
	}
	dstCtx, dstCancel := context.WithCancel(context.Background())
	defer dstCancel()
	dstDone := make(chan struct{})
	go func() {
		defer close(dstDone)
		w := io.Discard
		// w = os.Stderr
		CountFn(scripts.Count, c.Config, w, w, w)(dstCtx)
		CountFn(scripts.Count, c.Config, w, w, w)(dstCtx)
	}()
	<-dstDone

	if err := c.SameIndex(); err != nil {
		t.Fatalf("%+v", err)
	}
}
