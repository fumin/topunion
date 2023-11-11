package server

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/pkg/errors"

	"nvr"
	"nvr/util"
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
		t.Run(util.TimeFormat(tc.now), func(t *testing.T) {
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

			record := nvr.Record{ID: util.TimeFormat(tc.now)}
			camera0 := nvr.Camera{
				Name:  "testVid",
				Input: []string{"-stream_loop", "-1", "-re", "-i", testVid},
			}
			record.Camera = append(record.Camera, camera0)

			// Run src.
			recordDir := nvr.RecordDir(s.RecordDir, record.ID)
			camera0Dir := filepath.Join(recordDir, camera0.Name)
			if err := os.MkdirAll(camera0Dir, os.ModePerm); err != nil {
				t.Fatalf("%+v", err)
			}
			srcCtx, srcCancel := context.WithCancel(context.Background())
			defer srcCancel()
			srcDone := make(chan struct{})
			go func() {
				defer close(srcDone)
				getInput := func() ([]string, string, error) {
					input, err := camera0.GetInput()
					if err != nil {
						return nil, "", errors.Wrap(err, "")
					}
					return input, "239.0.0.1:10000", nil
				}
				nvr.RecordVideoFn(camera0Dir, getInput)(srcCtx)
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
