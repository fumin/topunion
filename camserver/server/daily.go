package server

import (
	"context"
	"database/sql"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/pkg/errors"

	"camserver"
	"camserver/util"
)

func DeleteOldVideoProcess(db *sql.DB) error {
	cutoff := time.Now().AddDate(0, 0, -1).Unix()
	deleteStr := `DELETE FROM ` + camserver.TableStatDedup + ` WHERE t < ?`
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if _, err := db.ExecContext(ctx, deleteStr, cutoff); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func DeleteOldVideos(root string, dur time.Duration) error {
	cutoff := time.Now().Add(-dur)
	cutoffDayStr := cutoff.Format(util.FormatDate)

	olds := make([]string, 0)
	cameras, err := os.ReadDir(root)
	if err != nil {
		return errors.Wrap(err, "")
	}
	for _, camera := range cameras {
		camDir := filepath.Join(root, camera.Name())
		camOlds := make([]string, 0, 30)
		years, err := os.ReadDir(camDir)
		if err != nil {
			return errors.Wrap(err, "")
		}
	CamLoop:
		for i := len(years) - 1; i >= 0; i-- {
			yearDir := filepath.Join(camDir, years[i].Name())
			days, err := os.ReadDir(yearDir)
			if err != nil {
				return errors.Wrap(err, "")
			}
			for j := len(days) - 1; j >= 0; j-- {
				day := days[j].Name()
				if day >= cutoffDayStr {
					continue
				}
				dayDir := filepath.Join(yearDir, day)
				camOlds = append(camOlds, dayDir)
				if len(camOlds) >= 30 {
					break CamLoop
				}
			}
		}
		olds = append(olds, camOlds...)
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
		if !isVideo(path) {
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

func isVideo(fpath string) bool {
	for _, videoExt := range []string{".ts", ".mp4"} {
		if filepath.Ext(fpath) == videoExt {
			return true
		}
	}
	return false
}
