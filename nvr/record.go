package nvr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/pkg/errors"

	"nvr/arp"
)

type RTSP struct {
	Name             string
	Link             string
	NetworkInterface string
	MacAddress       string
	Username         string
	Password         string
	Port             int
	Path             string

	Video string `json:",omitempty"`
}

func (info RTSP) GetLink() (string, error) {
	if info.Link != "" {
		return info.Link, nil
	}

	hws, err := arp.Scan(info.NetworkInterface)
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	hw, ok := hws[info.MacAddress]
	if !ok {
		return "", errors.Errorf("%#v %#v", info, hws)
	}
	info.Link = fmt.Sprintf("rtsp://%s:%s@%s:%d%s", info.Username, info.Password, hw.IP, info.Port, info.Path)

	return info.Link, nil
}

func (rtsp RTSP) Prepare(recordDir string) (string, error) {
	dir := filepath.Join(recordDir, rtsp.Name)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return "", errors.Wrap(err, "")
	}
	return dir, nil
}

type Track struct {
	Count int
}

type Count struct {
	Src    string
	Config CountConfig

	Track      *Track `json:",omitempty"`
	TrackVideo string `json:",omitempty"`
}

func (c Count) Fill(recordDir string) Count {
	c.Config.Src = filepath.Join(recordDir, c.Src, IndexM3U8)
	c.Config.TrackIndex = filepath.Join(recordDir, c.Src+"Track", IndexM3U8)
	c.Config.TrackDir = filepath.Join(filepath.Dir(c.Config.TrackIndex), "track")
	return c
}

func (c Count) Prepare() error {
	if err := os.MkdirAll(c.Config.TrackDir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	// Wait for src to appear.
	var err error
	for i := 0; i < HLSTime*4; i++ {
		_, err = os.Stat(c.Config.Src)
		if err == nil {
			break
		}
		<-time.After(time.Second)
	}
	if err != nil {
		return errors.Wrap(err, "")
	}

	return nil
}

func (c Count) SameIndex() error {
	src, err := os.ReadFile(c.Config.Src)
	if err != nil {
		return errors.Wrap(err, "")
	}
	track, err := os.ReadFile(c.Config.TrackIndex)
	if err != nil {
		return errors.Wrap(err, "")
	}
	if !bytes.Equal(src, track) {
		return errors.Errorf("not equal")
	}
	return nil
}

func (c Count) LastTrack() (*Track, error) {
	entries, err := os.ReadDir(c.Config.TrackDir)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	if len(entries) == 0 {
		return &Track{}, nil
	}

	last := filepath.Join(c.Config.TrackDir, entries[len(entries)-1].Name())
	b, err := os.ReadFile(last)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	var track Track
	if err := json.Unmarshal(b, &track); err != nil {
		return nil, errors.Wrap(err, "")
	}
	return &track, nil
}

type Record struct {
	ID    string
	RTSP  []RTSP
	Count []Count

	Err     string
	Create  time.Time
	Stop    time.Time
	Cleanup time.Time

	Link       string `json:",omitempty"`
	CreateTime string `json:",omitempty"`
	StopTime   string `json:",omitempty"`
}

func (r Record) Dir(root string) string {
	yearStr := r.ID[:4]
	dayStr := r.ID[:8]
	dir := filepath.Join(root, yearStr, dayStr, r.ID)
	return dir
}

func WriteRecord(root string, record Record) error {
	b, err := json.Marshal(record)
	if err != nil {
		return errors.Wrap(err, "")
	}

	recordDir := record.Dir(root)
	if err := os.MkdirAll(recordDir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	fpath := filepath.Join(recordDir, ValueFilename)
	if err := os.WriteFile(fpath, b, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func ReadRecord(root, id string) (Record, error) {
	if len(id) < 8 {
		return Record{}, errors.Errorf("%d", len(id))
	}
	yearStr := id[:4]
	dayStr := id[:8]
	fpath := filepath.Join(root, yearStr, dayStr, id, ValueFilename)
	b, err := os.ReadFile(fpath)
	if err != nil {
		return Record{}, errors.Wrap(err, "")
	}
	var record Record
	if err := json.Unmarshal(b, &record); err != nil {
		return Record{}, errors.Wrap(err, "")
	}
	return record, nil
}

func ListRecord(root string) ([]Record, error) {
	limit := 30
	latestIDs := make([]string, 0)
	years, err := os.ReadDir(root)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
Loop:
	for i := len(years) - 1; i >= 0; i-- {
		year := years[i].Name()
		days, err := os.ReadDir(filepath.Join(root, year))
		if err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("\"%s\"", year))
		}
		for j := len(days) - 1; j >= 0; j-- {
			day := days[j].Name()
			ids, err := os.ReadDir(filepath.Join(root, year, day))
			if err != nil {
				return nil, errors.Wrap(err, fmt.Sprintf("%s %s", year, day))
			}
			for k := len(ids) - 1; k >= 0; k-- {
				id := ids[k].Name()
				latestIDs = append(latestIDs, id)
				if len(latestIDs) >= limit {
					break Loop
				}
			}
		}
	}

	records := make([]Record, 0, len(latestIDs))
	for _, id := range latestIDs {
		record, err := ReadRecord(root, id)
		if err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("\"%s\"", id))
		}
		records = append(records, record)
	}
	return records, nil
}
