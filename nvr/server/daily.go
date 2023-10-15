package server

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/pkg/errors"
)

func (s *Server) DeleteOldVideos(dur time.Duration) error {
	cutoff := time.Now().Add(-dur)
	cutoffDayStr := cutoff.Format("20060102")

	olds := make([]string, 0)
	years, err := os.ReadDir(s.RecordDir)
	if err != nil {
		return errors.Wrap(err, "")
	}
Loop:
	for _, yearE := range years {
		year := yearE.Name()
		days, err := os.ReadDir(filepath.Join(s.RecordDir, year))
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("\"%s\"", year))
		}
		for _, dayE := range days {
			day := dayE.Name()
			if day >= cutoffDayStr {
				break Loop
			}
			olds = append(olds, filepath.Join(s.RecordDir, year, day))
		}
	}

	for _, old := range olds {
		if err := walkVideo(old, os.Remove); err != nil {
			return errors.Wrap(err, "")
		}
	}

	return nil
}

func walkVideo(dir string, fn func(string) error) error {
	err := fs.WalkDir(os.DirFS(dir), ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return errors.Wrap(err, "")
		}
		if d.IsDir() {
			return nil
		}
		if filepath.Ext(path) != ".ts" {
			return nil
		}
		if err := fn(filepath.Join(dir, path)); err != nil {
			return errors.Wrap(err, "")
		}
		return nil
	})
	if err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}
