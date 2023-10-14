package nvr

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestFFProbe(t *testing.T) {
	t.Parallel()
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer os.RemoveAll(dir)

	fpath := filepath.Join(dir, "test.ts")
	cmd := strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 -f mpegts "+fpath, " ")
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
}
