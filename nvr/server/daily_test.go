package server

import (
	"context"
	"io"
	"nvr"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDeleteOldVideos(t *testing.T) {
	t.Parallel()
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("%+v", err)
	}
	dir = "/tmp/273930960"
	// defer os.RemoveAll(dir)
	t.Logf("%s", dir)
	testVid := filepath.Join(dir, "test.mp4")
	cmd := strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 "+testVid, " ")
	if b, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		t.Fatalf("%+v %s", err, b)
	}
	s, err := NewServer(filepath.Join(dir, "server"), "")
	if err != nil {
		t.Fatalf("%+v", err)
	}

	nows := []time.Time{
		time.Now(),
		time.Date(2006, time.January, 2, 15, 4, 5, 0, time.UTC),
	}
	for _, now := range nows {
		record := nvr.Record{ID: nvr.TimeFormat(now)}
		rtsp0 := nvr.RTSP{Name: "testVid", Link: testVid}
		record.RTSP = append(record.RTSP, rtsp0)

		// Run src.
		recordDir := nvr.RecordDir(s.RecordDir, record.ID)
		if err := rtsp0.Prepare(recordDir); err != nil {
			t.Fatalf("%+v", err)
		}
		srcCtx, srcCancel := context.WithCancel(context.Background())
		defer srcCancel()
		go func() {
			nvr.RecordVideoFn(rtsp0.Dir(recordDir), rtsp0.GetLink, io.Discard, io.Discard, io.Discard)(srcCtx)
		}()
		<-time.After(500 * time.Millisecond)
		srcCancel()
	}

	if err := s.DeleteOldVideos(1); err != nil {
		t.Fatalf("%+v", err)
	}
}
