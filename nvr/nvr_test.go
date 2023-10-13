package nvr

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const (
	trackSubdir = "track"
)

func TestCount(t *testing.T) {
	t.Parallel()
	// dir, err := os.MkdirTemp("", "")
	// if err != nil {
	// 	t.Fatalf("%+v", err)
	// }
	// defer os.RemoveAll(dir)
	dir := "mytest"
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		t.Fatalf("%+v", err)
	}

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

	count := countCmd(dst, src, scripts.Count)
	// Write the discontinuity after 0.ts.
	stdouterr, err := exec.Command(count[0], count[1:]...).CombinedOutput()
	if err != nil {
		t.Fatalf("%+v \"%s\" %s", err, strings.Join(count, " "), stdouterr)
	}
	// Write till the end.
	stdouterr, err = exec.Command(count[0], count[1:]...).CombinedOutput()
	if err != nil {
		t.Fatalf("%+v \"%s\" %s", err, strings.Join(count, " "), stdouterr)
	}
	t.Logf("%s", stdouterr)
}

func countCmd(dst, src, script string) []string {
	var config CountConfig
	config.TrackIndex = dst
	config.TrackDir = filepath.Join(filepath.Dir(dst), trackSubdir)
	config.Src = src
	config.AI.Smart = false
	config.AI.Device = "cpu"
	config.AI.Mask.Enable = false
	config.AI.Yolo.Weights = "yolo_best.pt"
	config.AI.Yolo.Size = 640
	cfg, err := json.Marshal(config)
	if err != nil {
		panic(err)
	}
	return []string{"python", script, "-c=" + string(cfg)}
}
