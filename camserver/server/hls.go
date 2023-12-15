package server

import (
	"camserver"
	"os"
	"path"
	"path/filepath"
	"slices"

	"github.com/pkg/errors"

	"camserver/hls"
	"camserver/util"
)

func getPlaylist(root, filename string) (hls.Playlist, error) {
	segs, err := os.ReadDir(root)
	if err != nil {
		return hls.Playlist{}, errors.Wrap(err, "")
	}

	var pl hls.Playlist
	for i := len(segs) - 1; i >= 0; i-- {
		segDE := segs[i]

		if len(pl.Segment) > 5 {
			break
		}
		segDir := filepath.Join(root, segDE.Name())
		rawDir := filepath.Join(segDir, camserver.RawDir)
		doneRaw, err := util.GetDoneTry(rawDir)
		if err != nil {
			continue
		}
		pvDir := filepath.Join(segDir, camserver.ProcessVideoDir)
		doneProcess, err := util.GetDoneTry(pvDir)
		if err != nil {
			continue
		}

		var seg hls.Segment
		rawDone := filepath.Join(rawDir, doneRaw, util.DoneFilename)
		if err := util.ReadJSONFile(rawDone, &seg); err != nil {
			continue
		}
		seg.URL = path.Join(pvDir, doneProcess, filename)

		pl.MediaSequence = i
		pl.Segment = append(pl.Segment, seg)
	}

	// Sort segments ascending.
	slices.Reverse(pl.Segment)

	return pl, nil
}

func latestDay(root string) ([2]string, error) {
	years, err := os.ReadDir(root)
	if err != nil {
		return [2]string{}, errors.Wrap(err, "")
	}
	for i := len(years) - 1; i >= 0; i-- {
		year := years[i]

		yearDir := filepath.Join(root, year.Name())
		days, err := os.ReadDir(yearDir)
		if err != nil {
			return [2]string{}, errors.Wrap(err, "")
		}
		for j := len(days) - 1; j >= 0; j-- {
			day := days[j]
			return [2]string{year.Name(), day.Name()}, nil
		}
	}
	return [2]string{}, errors.Errorf("not found")
}
