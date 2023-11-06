package server

import (
	"context"
	"nvr"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/pkg/errors"
)

func TestDeleteOldVideos(t *testing.T) {
	tests := []struct {
		now     time.Time
		deleted bool
	}{
		{now: time.Now(), deleted: false},
		{now: time.Date(2006, time.January, 2, 15, 4, 5, 0, time.UTC), deleted: true},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(nvr.TimeFormat(tc.now), func(t *testing.T) {
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

			record := nvr.Record{ID: nvr.TimeFormat(tc.now)}
			rtsp0 := nvr.RTSP{
				Name:  "testVid",
				Input: []string{"-stream_loop", "-1", "-re", "-i", testVid},
			}
			record.RTSP = append(record.RTSP, rtsp0)

			// Run src.
			recordDir := nvr.RecordDir(s.RecordDir, record.ID)
			if err := rtsp0.Prepare(recordDir); err != nil {
				t.Fatalf("%+v", err)
			}
			srcCtx, srcCancel := context.WithCancel(context.Background())
			defer srcCancel()
			srcDone := make(chan struct{})
			go func() {
				defer close(srcDone)
				nvr.RecordVideoFn(rtsp0.Dir(recordDir), rtsp0.GetInput)(srcCtx)
			}()
			for i := 0; i < 10; i++ {
				n, _ := numVideos(recordDir)
				if n > 0 {
					break
				}
				<-time.After(100 * time.Millisecond)
			}
			srcCancel()
			<-srcDone

			if err := s.DeleteOldVideos(24 * time.Hour); err != nil {
				t.Fatalf("%+v", err)
			}

			n, err := numVideos(recordDir)
			if err != nil {
				t.Fatalf("%+v", err)
			}
			if (n == 0) != tc.deleted {
				t.Fatalf("%d %v", n, tc.deleted)
			}
		})
	}
}

func numVideos(dir string) (int, error) {
	n := 0
	err := walkVideo(dir, func(string) error {
		n++
		return nil
	})
	if err != nil {
		return -1, errors.Wrap(err, "")
	}
	return n, nil
}
