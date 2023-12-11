package camserver

import (
	"camserver/cuda"
	"camserver/ffmpeg"
	"camserver/util"
	"context"
	"database/sql"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"
)

func TestProcessVideo(t *testing.T) {
	t.Parallel()
	env := newEnvironment(t)
	defer env.close()

	cfg := shilinSDConfig()
	counter, err := NewCounter(env.scripts.Count, cfg)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer counter.Close()

	testVid := filepath.Join(env.dir, "test.mp4")
	if err := util.CopyFile(testVid, filepath.Join("testing", "shilin20230826_sd.mp4")); err != nil {
		t.Fatalf("%+v", err)
	}

	arg := ProcessVideoInput{
		Camera:   "testcam",
		Dir:      env.dir,
		Filepath: testVid,
		Time:     time.Date(2023, time.December, 2, 1, 43, 22, 0, util.TaipeiTZ),
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if err := ProcessVideo(ctx, env.db, counter, arg); err != nil {
		t.Fatalf("%+v", err)
	}

	// Check mpegts output.
	runDir, err := lastDirEntry(filepath.Join(env.dir, ProcessVideoDir))
	if err != nil {
		t.Fatalf("%+v", err)
	}
	mpegts := filepath.Join(runDir, RawMPEGTSFilename)
	probe, err := ffmpeg.FFProbe(ctx, []string{mpegts})
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if probe.Format.Duration != 17.1 {
		t.Fatalf("%+v", probe)
	}

	// Check count.
	qt := time.Date(2023, time.December, 1, 17, 0, 0, 0, time.UTC)
	n, err := readStat(ctx, env.db, qt, arg.Camera)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if n != 10 {
		t.Fatalf("%d", n)
	}
}

func lastDirEntry(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	if len(entries) == 0 {
		return "", errors.Errorf("empty dir")
	}
	last := entries[len(entries)-1]
	return filepath.Join(dir, last.Name()), nil
}

func shilinSDConfig() CountConfig {
	device := "cpu"
	if cuda.IsAvailable() {
		device = "cuda:0"
	}
	cfg := CountConfig{Height: 480, Width: 640, Device: device}
	cfg.Mask.Enable = true
	cfg.Mask.Crop.W = 999999
	cfg.Mask.Mask.Slope = 5
	cfg.Mask.Mask.Y = 160
	cfg.Mask.Mask.H = 70
	cfg.Yolo.Weights = "yolo_best.pt"
	cfg.Yolo.Size = 640
	return cfg
}

type environment struct {
	dir     string
	db      *sql.DB
	scripts Scripts
}

func newEnvironment(t *testing.T) *environment {
	dir, err := os.MkdirTemp("", t.Name())
	if err != nil {
		t.Fatalf("%+v", err)
	}
	env := &environment{dir: dir}

	dbPath := filepath.Join(env.dir, "db.sqlite")
	dbV := url.Values{}
	dbV.Set("_journal_mode", "WAL")
	env.db, err = sql.Open("sqlite3", "file:"+dbPath+"?"+dbV.Encode())
	if err != nil {
		t.Fatalf("%+v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := CreateTables(ctx, env.db); err != nil {
		t.Fatalf("%+v", err)
	}

	env.scripts, err = NewScripts(dir)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	return env
}

func (env *environment) close() {
	env.db.Close()
	os.RemoveAll(env.dir)
}
