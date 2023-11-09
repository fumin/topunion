package nvr

import (
	"context"
	"database/sql"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"nvr/cuda"
	"nvr/util"
)

func TestInsertRecord(t *testing.T) {
	t.Parallel()
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer os.RemoveAll(dir)
	dbPath := filepath.Join(dir, "db.sqlite")
	db, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer db.Close()
	if err := CreateTables(db); err != nil {
		t.Fatalf("%+v", err)
	}

	r, err := InsertRecord(db, "", time.Now(), Record{})
	if err != nil {
		t.Fatalf("%+v", err)
	}
	readR, err := GetRecord(db, r.ID)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if readR.ID != r.ID {
		t.Fatalf("%#v %#v", readR, r)
	}
	if readR.Create.UnixNano() != r.Create.UnixNano() {
		t.Fatalf("%#v %#v", readR, r)
	}
	if readR.Stop.UnixNano() != r.Stop.UnixNano() {
		t.Fatalf("%#v %#v", readR, r)
	}
}

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
	getInput := func() ([]string, string, error) {
		// Just use some multicast address, since we do not care about the video content.
		const multicastAddr = "239.0.0.1:10000"
		return []string{"-stream_loop", "-1", "-re", "-i", testVid}, multicastAddr, nil
	}
	srcCtx, srcCancel := context.WithCancel(context.Background())
	defer srcCancel()
	go func() {
		RecordVideoFn(srcDir, getInput)(srcCtx)
	}()

	// Run dst.
	c := Count{Src: filepath.Base(srcDir)}
	c.Config.AI.Smart = cuda.IsAvailable()
	c.Config.AI.Device = "cpu"
	if c.Config.AI.Smart {
		c.Config.AI.Device = "cuda:0"
	}
	c.Config.AI.Yolo.Weights = "yolo_best.pt"
	c.Config.AI.Yolo.Size = 640
	c = c.Fill(dir)
	if err := c.Prepare(); err != nil {
		b, _ := os.ReadFile(filepath.Join(srcDir, util.StderrFilename))
		t.Logf("%s", b)
		t.Fatalf("%+v", err)
	}
	dstDir := filepath.Dir(c.Config.TrackIndex)
	if err := os.MkdirAll(dstDir, os.ModePerm); err != nil {
		t.Fatalf("%+v", err)
	}
	dstCtx, dstCancel := context.WithCancel(context.Background())
	defer dstCancel()
	dstDone := make(chan struct{})
	go func() {
		defer close(dstDone)
		CountFn(dstDir, scripts.Count, c.Config)(dstCtx)
		CountFn(dstDir, scripts.Count, c.Config)(dstCtx)
	}()

	<-time.After(500 * time.Millisecond)
	srcCancel()
	<-dstDone

	if err := c.SameIndex(); err != nil {
		t.Fatalf("%+v", err)
	}
}
