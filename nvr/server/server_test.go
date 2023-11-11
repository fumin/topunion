package server

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/pkg/errors"

	"nvr"
	"nvr/cuda"
	"nvr/ffmpeg"
	"nvr/util"
)

func TestEgg(t *testing.T) {
	t.Parallel()
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	dir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewServer(filepath.Join(dir, "server"), "", "239.0.0.16/28")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer s.Close()
	if err := nvr.CreateTables(s.DB); err != nil {
		t.Fatalf("%+v", err)
	}

	var record nvr.Record
	video := filepath.Join("../", "sample", "shilin20230826.mp4")
	camera0 := nvr.Camera{Name: "camera0", Input: []string{video}, Repeat: 2}
	record.Camera = append(record.Camera, camera0)

	count0 := nvr.Count{Src: camera0.Name}
	count0.Config.AI.Smart = true
	count0.Config.AI.Device = "cuda:0"
	count0.Config.AI.Mask.Enable = true
	count0.Config.AI.Mask.Crop.X = 100
	count0.Config.AI.Mask.Crop.Y = 0
	count0.Config.AI.Mask.Crop.W = 1700
	count0.Config.AI.Mask.Mask.Slope = 10
	count0.Config.AI.Mask.Mask.Y = 500
	count0.Config.AI.Mask.Mask.H = 200
	count0.Config.AI.Yolo.Weights = filepath.Join("../", "yolo_best.pt")
	count0.Config.AI.Yolo.Size = 640
	record.Count = append(record.Count, count0)

	id, err := s.startRecord(record)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	// Collect tracks.
	tracks := make([]nvr.Track, 0)
	for {
		r, err := nvr.GetRecord(s.DB, id)
		if err == nil {
			t := r.Count[0].Track
			if len(tracks) == 0 || tracks[len(tracks)-1] != t {
				tracks = append(tracks, t)
			}
		}
		s.records.RLock()
		numRunning := len(s.records.m)
		s.records.RUnlock()
		if numRunning == 0 {
			break
		}
		<-time.After(time.Second)
	}

	// Check track history is correct.
	if !(len(tracks) > 1) {
		t.Fatalf("%#v", tracks)
	}
	if !(tracks[(len(tracks)-1)/2].Count < tracks[len(tracks)-1].Count) {
		t.Fatalf("%#v", tracks)
	}
	if tracks[len(tracks)-1].Count != 49*camera0.Repeat {
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
	s, err := NewServer(filepath.Join(dir, "server"), "", "239.0.0.0/28")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	defer s.Close()
	if err := nvr.CreateTables(s.DB); err != nil {
		t.Fatalf("%+v", err)
	}

	record := nvr.Record{ID: util.TimeFormat(time.Now())}
	camera0 := nvr.Camera{
		Name:  "testVid",
		Input: []string{"-stream_loop", "-1", "-re", "-i", testVid},
	}
	record.Camera = append(record.Camera, camera0)
	count0 := nvr.Count{Src: camera0.Name}
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

	var videoSecs float64 = 10
	<-time.After(time.Duration(videoSecs*1e9) * time.Nanosecond)

	if err := s.records.cancel(id); err != nil {
		t.Fatalf("%+v", err)
	}
	// Wait at most a few seconds for the record to clean up.
	var readRecord nvr.Record
	for i := 0; ; i++ {
		readRecord, err = func() (nvr.Record, error) {
			r, err := nvr.GetRecord(s.DB, id)
			if err != nil {
				return nvr.Record{}, errors.Wrap(err, "")
			}
			if r.Cleanup.IsZero() {
				return nvr.Record{}, errors.Errorf("zero %#v", r)
			}
			return r, nil
		}()
		if err == nil || i > 30 {
			break
		}
		<-time.After(time.Second)
	}
	if err != nil {
		b, _ := os.ReadFile(filepath.Join(filepath.Dir(record.Count[0].Config.TrackIndex), util.StderrFilename))
		t.Logf("%s", b)
		t.Fatalf("%+v", err)
	}

	// Check video output.
	probe, err := ffmpeg.FFProbe([]string{"-i", readRecord.Count[0].Config.TrackIndex})
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if math.Abs(probe.Format.Duration/videoSecs-1) > 0.1 {
		t.Fatalf("%#v", probe)
	}

	// Check child processes exited with exit code 0.
	msgs, err := util.ReadCmdMsg(filepath.Join(filepath.Dir(readRecord.Count[0].Config.TrackIndex), util.StatusFilename))
	if err != nil {
		t.Fatalf("%+v", err)
	}
	exits := make([]util.CmdMsg, 0)
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
