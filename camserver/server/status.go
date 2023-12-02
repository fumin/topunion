package server

import (
	"camserver"
	"camserver/util"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/pkg/errors"
)

type CameraStatus struct {
	ID        string
	Latest    time.Time
	LatestErr string
}

type StatusPage struct {
	Camera []CameraStatus
}

func (s *Server) getStatusPage() (StatusPage, error) {
	camIDs := make([]string, 0, len(s.Camera))
	for id := range s.Camera {
		camIDs = append(camIDs, id)
	}
	slices.Sort(camIDs)

	var page StatusPage
	for _, id := range camIDs {
		c := CameraStatus{ID: id}

		camDir := filepath.Join(s.VideoDir, id)
		var err error
		c.Latest, err = getCameraLatest(camDir)
		if err != nil {
			c.LatestErr = fmt.Sprintf("%+v", err)
		}
		c.Latest = c.Latest.In(util.TaipeiTZ)

		page.Camera = append(page.Camera, c)
	}
	return page, nil
}

func getCameraLatest(dir string) (time.Time, error) {
	years, err := os.ReadDir(dir)
	if err != nil {
		return time.Time{}, errors.Wrap(err, "")
	}
	if len(years) == 0 {
		return time.Time{}, errors.Errorf("not found")
	}
	latestYear := filepath.Join(dir, years[len(years)-1].Name())

	days, err := os.ReadDir(latestYear)
	if err != nil {
		return time.Time{}, errors.Wrap(err, "")
	}
	if len(days) == 0 {
		return time.Time{}, errors.Errorf("not found")
	}
	latestDay := filepath.Join(latestYear, days[len(days)-1].Name())

	videos, err := os.ReadDir(latestDay)
	if err != nil {
		return time.Time{}, errors.Wrap(err, "")
	}
	for i := len(videos) - 1; i >= 0; i-- {
		raw := filepath.Join(latestDay, videos[i].Name(), camserver.RawDir)
		runs, err := os.ReadDir(raw)
		if err != nil {
			return time.Time{}, errors.Wrap(err, "")
		}
		for j := len(runs) - 1; j >= 0; j-- {
			runID := runs[j].Name()
			t, err := util.ParseRunTime(runID)
			if err != nil {
				return time.Time{}, errors.Wrap(err, "")
			}
			done := filepath.Join(raw, runID, util.DoneFilename)
			if _, err := os.Stat(done); err == nil {
				return t, nil
			}
		}
	}
	return time.Time{}, errors.Errorf("not found")
}
