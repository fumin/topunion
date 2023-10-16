package nvr

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
)

func TestFFProbe(t *testing.T) {
	t.Parallel()
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer os.RemoveAll(dir)

	var videoSecs float64 = 1
	fpath := filepath.Join(dir, "test.ts")
	cmd := []string{"ffmpeg", "-f", "lavfi", "-i", "smptebars", "-t", strconv.FormatFloat(videoSecs, 'f', -1, 64), "-f", "mpegts", fpath}
	if b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		t.Fatalf("%+v %s", err, b)
	}

	probe, err := FFProbe(fpath)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if probe.Format.FormatName != "mpegts" {
		t.Fatalf("%#v", probe)
	}
	if math.Abs(probe.Format.Duration-videoSecs) > 0.05 {
		t.Fatalf("%#v", probe)
	}
}
