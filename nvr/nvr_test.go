package nvr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"nvr/cuda"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/pkg/errors"
)

const (
	trackSubdir = "track"
)

func TestCountDiscontinuity(t *testing.T) {
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

	srcDir := filepath.Join(dir, "src")
	if err := os.MkdirAll(srcDir, os.ModePerm); err != nil {
		t.Fatalf("%+v", err)
	}
	cmd := strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 -f mpegts "+filepath.Join(srcDir, "0.ts"), " ")
	if b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		t.Fatalf("%+v %s", err, b)
	}
	cmd = strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 -f mpegts "+filepath.Join(srcDir, "1.ts"), " ")
	if b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		t.Fatalf("%+v %s", err, b)
	}
	b := `#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:1
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-DISCONTINUITY
#EXTINF:0.1,
0.ts
#EXT-X-DISCONTINUITY
#EXTINF:0.1,
1.ts
#EXT-X-ENDLIST`
	src := filepath.Join(srcDir, IndexM3U8)
	if err := os.WriteFile(src, []byte(b), os.ModePerm); err != nil {
		t.Fatalf("%+v %s", err, b)
	}

	dstDir := filepath.Join(dir, "dst")
	trackDir := filepath.Join(dstDir, trackSubdir)
	if err := os.MkdirAll(trackDir, os.ModePerm); err != nil {
		t.Fatalf("%+v", err)
	}
	cmd = strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 -f mpegts "+filepath.Join(dstDir, "0.ts"), " ")
	if b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		t.Fatalf("%+v %s", err, b)
	}
	if err := os.WriteFile(filepath.Join(trackDir, "0.json"), []byte(`{"Count": 0}`), os.ModePerm); err != nil {
		t.Fatalf("%+v", err)
	}
	b = `#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:1
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-DISCONTINUITY
#EXTINF:0.1,
0.ts
#EXT`
	dst := filepath.Join(dstDir, IndexM3U8)
	if err := os.WriteFile(dst, []byte(b), os.ModePerm); err != nil {
		t.Fatalf("%+v %s", err, b)
	}

	// Write the discontinuity after 0.ts.
	_, _, err = runCount(dst, src, scripts.Count)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	// Write till the end.
	logs, _, err := runCount(dst, src, scripts.Count)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if err := lastResultEq(filepath.Join(trackDir, "0.json"), logs); err != nil {
		t.Fatalf("%+v", err)
	}
	if err := warmupEq("", logs); err != nil {
		t.Fatalf("%+v", err)
	}
}

func TestCountWarmup(t *testing.T) {
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

	srcDir := filepath.Join(dir, "src")
	if err := os.MkdirAll(srcDir, os.ModePerm); err != nil {
		t.Fatalf("%+v", err)
	}
	cmd := strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 -f mpegts "+filepath.Join(srcDir, "0.ts"), " ")
	if b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		t.Fatalf("%+v %s", err, b)
	}
	cmd = strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 -f mpegts "+filepath.Join(srcDir, "1.ts"), " ")
	if b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		t.Fatalf("%+v %s", err, b)
	}
	b := `#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:1
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-DISCONTINUITY
#EXTINF:0.1,
0.ts
#EXTINF:0.1,
1.ts
#EXT-X-ENDLIST`
	src := filepath.Join(srcDir, IndexM3U8)
	if err := os.WriteFile(src, []byte(b), os.ModePerm); err != nil {
		t.Fatalf("%+v %s", err, b)
	}

	dstDir := filepath.Join(dir, "dst")
	trackDir := filepath.Join(dstDir, trackSubdir)
	if err := os.MkdirAll(trackDir, os.ModePerm); err != nil {
		t.Fatalf("%+v", err)
	}
	cmd = strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 -f mpegts "+filepath.Join(dstDir, "0.ts"), " ")
	if b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		t.Fatalf("%+v %s", err, b)
	}
	if err := os.WriteFile(filepath.Join(trackDir, "0.json"), []byte(`{"Count": 0}`), os.ModePerm); err != nil {
		t.Fatalf("%+v", err)
	}
	b = `#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:1
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-DISCONTINUITY
#EXTINF:0.1,
0.ts`
	dst := filepath.Join(dstDir, IndexM3U8)
	if err := os.WriteFile(dst, []byte(b), os.ModePerm); err != nil {
		t.Fatalf("%+v %s", err, b)
	}

	// Write till the end.
	logs, stdouterr, err := runCount(dst, src, scripts.Count)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if err := lastResultEq(filepath.Join(trackDir, "0.json"), logs); err != nil {
		t.Fatalf("%+v %s", err, stdouterr)
	}
	if err := warmupEq(filepath.Join(srcDir, "0.ts"), logs); err != nil {
		t.Fatalf("%+v %s", err, stdouterr)
	}
}

func lastResultEq(expected string, logs []map[string]interface{}) error {
	for _, l := range logs {
		if l["Levent"] != "lastResult" {
			continue
		}
		if l["V"] != expected {
			return errors.Errorf("\"%s\"", l["V"])
		}
		return nil
	}
	return errors.Errorf("not found")
}

func warmupEq(expected string, logs []map[string]interface{}) error {
	for _, l := range logs {
		if l["Levent"] != "warmup" {
			continue
		}
		if l["V"] != expected {
			return errors.Errorf("\"%s\"", l["V"])
		}
		return nil
	}
	return errors.Errorf("not found")
}

func runCount(dst, src, script string) ([]map[string]interface{}, []byte, error) {
	cmd := countCmd(dst, src, script)
	// Write the discontinuity after 0.ts.
	b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput()
	if err != nil {
		return nil, nil, errors.Wrap(err, fmt.Sprintf("\"%s\" %s", strings.Join(cmd, " "), b))
	}

	logs := make([]map[string]interface{}, 0)
	dec := json.NewDecoder(bytes.NewBuffer(b))
	for {
		var m map[string]interface{}
		err := dec.Decode(&m)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, errors.Wrap(err, "")
		}
		logs = append(logs, m)
	}
	return logs, b, nil
}

func countCmd(dst, src, script string) []string {
	var config CountConfig
	config.TrackIndex = dst
	config.TrackDir = filepath.Join(filepath.Dir(dst), trackSubdir)
	config.Src = src
	config.AI.Smart = cuda.IsAvailable()
	config.AI.Device = "cpu"
	config.AI.Mask.Enable = false
	config.AI.Yolo.Weights = "yolo_best.pt"
	config.AI.Yolo.Size = 640
	cfg, err := json.Marshal(config)
	if err != nil {
		panic(err)
	}
	return []string{Python, script, "-c=" + string(cfg)}
}
