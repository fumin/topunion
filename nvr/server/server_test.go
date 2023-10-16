package server

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nvr"
)

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

	record := nvr.Record{ID: nvr.TimeFormat(time.Now())}
	rtsp0 := nvr.RTSP{Name: "testVid", Link: testVid}
	record.RTSP = append(record.RTSP, rtsp0)
	count0 := nvr.Count{Src: rtsp0.Name}
	count0.Config.AI.Smart = false
	count0.Config.AI.Device = "cpu"
	count0.Config.AI.Mask.Enable = false
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
	var readErr error
	for i := 0; i < 30; i++ {
		readRecord, readErr = nvr.ReadRecord(s.RecordDir, id)
		if readErr == nil && !readRecord.Cleanup.IsZero() {
			break
		}
		<-time.After(time.Second)
	}
	if readErr != nil {
		t.Fatalf("%+v", err)
	}
	if readRecord.Cleanup.IsZero() {
		t.Fatalf("%#v", readRecord)
	}

	// Check video output.
	probe, err := nvr.FFProbe(readRecord.Count[0].Config.TrackIndex)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if math.Abs(probe.Format.Duration-videoSecs) > 0.1 {
		t.Fatalf("%#v", probe)
	}

	// Check child processes exited with exit code 0.
	msgs, err := nvr.ReadCmdMsg(filepath.Join(filepath.Dir(readRecord.Count[0].Config.TrackIndex), statusFilename))
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
