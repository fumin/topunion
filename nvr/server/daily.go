package server

import (
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/pkg/errors"
)

func (s *Server) DeleteOldVideos(days int) error {
	cutoff := time.Now().Add(-time.Duration(days) * 24 * time.Hour)
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
	log.Printf("vvvvvvv %#v", olds)

	for _, old := range olds {
		matches, err := fs.Glob(os.DirFS(old), "*")
		if err != nil {
			return errors.Wrap(err, "")
		}
		log.Printf("%s %#v", old, matches)
		for _, m := range matches {
			log.Printf("%#v", m)
		}
	}

	return nil
}
