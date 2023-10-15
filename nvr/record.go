package nvr

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
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
	var err error
	for i := 0; i < 10; i++ {
		var link string
		link, err = info.getLink()
		if err == nil {
			return link, nil
		}
		<-time.After(500 * time.Millisecond)
	}
	return "", err
}

func (info RTSP) getLink() (string, error) {
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

func (rtsp RTSP) Dir(recordDir string) string {
	return filepath.Join(recordDir, rtsp.Name)
}

func (rtsp RTSP) Prepare(recordDir string) error {
	link, err := rtsp.GetLink()
	if err != nil {
		return errors.Wrap(err, "")
	}
	if _, err := FFProbe(link); err != nil {
		return errors.Wrap(err, "")
	}

	if err := os.MkdirAll(rtsp.Dir(recordDir), os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
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
	waitSecs := HLSTime * 16
	var err error
	for i := 0; i < waitSecs; i++ {
		_, err = os.Stat(c.Config.Src)
		if err == nil {
			break
		}
		<-time.After(time.Second)
	}
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf("no src after %d seconds", waitSecs))
	}

	return nil
}

func (c Count) SameIndex() error {
	srcF, err := os.Open(c.Config.Src)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer srcF.Close()
	trackF, err := os.Open(c.Config.TrackIndex)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer trackF.Close()

	src, track := bufio.NewScanner(srcF), bufio.NewScanner(trackF)
	for src.Scan() {
		if !track.Scan() {
			return errors.Errorf("no track")
		}
		if src.Text() != track.Text() {
			return errors.Errorf("\"%s\" \"%s\"", src.Text(), track.Text())
		}
	}
	if track.Scan() {
		return errors.Errorf("track not EOF")
	}
	if err := src.Err(); err != nil {
		return errors.Wrap(err, "")
	}
	if err := track.Err(); err != nil {
		return errors.Wrap(err, "")
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

func RecordDir(root, id string) string {
	yearStr := id[:4]
	dayStr := id[:8]
	dir := filepath.Join(root, yearStr, dayStr, id)
	return dir
}

func WriteRecord(root string, record Record) error {
	b, err := json.Marshal(record)
	if err != nil {
		return errors.Wrap(err, "")
	}

	recordDir := RecordDir(root, record.ID)
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
	fpath := filepath.Join(RecordDir(root, id), ValueFilename)
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

func cleanup(root, id string) error {
	t, err := TimeParse(id)
	if err == nil && t.Before(time.Now().AddDate(0, 0, -1)) {
		return nil
	}

	dir := RecordDir(root, id)
	log.Printf("cleaning up %s", dir)
	if err := os.RemoveAll(dir); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func ListRecord(root string) ([]Record, error) {
	limit := 30
	records := make([]Record, 0)
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
				r, err := ReadRecord(root, id)
				if err != nil {
					cleanup(root, id)
					continue
				}
				records = append(records, r)
				if len(records) >= limit {
					break Loop
				}
			}
		}
	}

	return records, nil
}
