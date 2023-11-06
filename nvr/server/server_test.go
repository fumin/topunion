package server

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"nvr/cuda"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nvr"

	"github.com/pkg/errors"
)

func TestEgg(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	t.Parallel()
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewServer(filepath.Join(dir, "server"), "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if err := createTables(s.db); err != nil {
		t.Fatalf("%+v", err)
	}

	var record nvr.Record
	video := filepath.Join("sample", "shilin20230826.mp4")
	rtsp0 := nvr.RTSP{Name: "rtsp0", Input: []string{video}, Repeat: 1}
	record.RTSP = append(record.RTSP, rtsp0)

	count0 := nvr.Count{Src: rtsp0.Name}
	count0.Config.AI.Smart = true
	count0.Config.AI.Device = "cuda:0"
	count0.Config.AI.Mask.Enable = true
	count0.Config.AI.Mask.Crop.X = 100
	count0.Config.AI.Mask.Crop.Y = 0
	count0.Config.AI.Mask.Crop.W = 1700
	count0.Config.AI.Mask.Mask.Slope = 10
	count0.Config.AI.Mask.Mask.Y = 500
	count0.Config.AI.Mask.Mask.H = 200
	count0.Config.AI.Yolo.Weights = "yolo_best.pt"
	count0.Config.AI.Yolo.Size = 640
	record.Count = append(record.Count, count0)

	id, err := s.startRecord(record)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	// Collect tracks.
	tracks := make([]nvr.Track, 0)
	for {
		r, err := nvr.GetRecord(s.db, id)
		if err == nil {
			t := r.Count[0].Track
			if len(tracks) > 0 && tracks[len(tracks)-1] != t {
				tracks = append(tracks, t)
			}
		}
		if len(s.records.all()) == 0 {
			break
		}
		<-time.After(time.Second)
	}

	// Check track history is correct.
	if len(tracks) < 2 {
		t.Fatalf("%#v", tracks)
	}
	if tracks[(len(tracks)-1)/2].Count >= tracks[len(tracks)-1].Count {
		t.Fatalf("%#v", tracks)
	}
	if tracks[len(tracks)-1].Count != 98 {
		t.Fatalf("%#v", tracks)
	}
}

func TestStartRecord(t *testing.T) {
	t.Parallel()
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer os.RemoveAll(dir)
	testVid := filepath.Join(dir, "test.mp4")
	cmd := strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 "+testVid, " ")
	if b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		t.Fatalf("%+v %s", err, b)
	}
	s, err := NewServer(filepath.Join(dir, "server"), "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if err := createTables(s.db); err != nil {
		t.Fatalf("%+v", err)
	}

	record := nvr.Record{ID: nvr.TimeFormat(time.Now())}
	rtsp0 := nvr.RTSP{
		Name:  "testVid",
		Input: []string{"-stream_loop", "-1", "-re", "-i", testVid},
	}
	record.RTSP = append(record.RTSP, rtsp0)
	count0 := nvr.Count{Src: rtsp0.Name}
	count0.Config.AI.Smart = cuda.IsAvailable()
	count0.Config.AI.Device = "cpu"
	if count0.Config.AI.Smart {
		count0.Config.AI.Device = "cuda:0"
	}
	count0.Config.AI.Mask.Enable = false
	count0.Config.AI.Yolo.Weights = "../yolo_best.pt"
	count0.Config.AI.Yolo.Size = 640
	record.Count = append(record.Count, count0)

	id, err := s.startRecord(record)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	var videoSecs float64 = 2
	<-time.After(time.Duration(videoSecs*1e9) * time.Nanosecond)

	if err := s.stopRecord(id); err != nil {
		t.Fatalf("%+v", err)
	}
	// Wait at most a few seconds for the record to clean up.
	var readRecord nvr.Record
	for i := 0; ; i++ {
		readRecord, err = func() (nvr.Record, error) {
			records, err := nvr.SelectRecord(s.db, "WHERE id=?", []interface{}{id})
			if err != nil {
				return nvr.Record{}, errors.Wrap(err, "")
			}
			if len(records) == 0 {
				return nvr.Record{}, errors.Errorf("not found")
			}
			if records[0].Cleanup.IsZero() {
				return nvr.Record{}, errors.Errorf("zero %#v", records[0])
			}
			return records[0], nil
		}()
		if err == nil || i > 30 {
			break
		}
		<-time.After(time.Second)
	}
	if err != nil {
		b, _ := os.ReadFile(filepath.Join(filepath.Dir(record.Count[0].Config.TrackIndex), nvr.StderrFilename))
		t.Logf("%s", b)
		t.Fatalf("%+v", err)
	}

	// Check video output.
	probe, err := nvr.FFProbe([]string{"-i", readRecord.Count[0].Config.TrackIndex})
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if math.Abs(probe.Format.Duration-videoSecs) > 0.1 {
		t.Fatalf("%#v", probe)
	}

	// Check child processes exited with exit code 0.
	msgs, err := nvr.ReadCmdMsg(filepath.Join(filepath.Dir(readRecord.Count[0].Config.TrackIndex), nvr.StatusFilename))
	if err != nil {
		t.Fatalf("%+v", err)
	}
	exits := make([]nvr.CmdMsg, 0)
	for _, m := range msgs {
		if m.ExitCode != nil {
			exits = append(exits, m)
		}
	}
	if len(exits) == 0 {
		t.Fatalf("no exits")
	}
	for _, m := range exits {
		if *m.ExitCode != 0 {
			t.Fatalf("%+v", m)
		}
	}
}

func createTables(db *sql.DB) error {
	sqlStrs := []string{
		`CREATE TABLE record (
			id text PRIMARY KEY,
			rtsp text,
			count text,
			err text,
			createAt datetime,
			stop datetime,
			cleanup datetime,
			track text);`,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for _, sqlStr := range sqlStrs {
		if _, err := db.ExecContext(ctx, sqlStr); err != nil {
			return errors.Wrap(err, fmt.Sprintf("\"%s\"", sqlStr))
		}
	}
	return nil
}
