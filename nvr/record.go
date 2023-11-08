package nvr

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"html/template"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/pkg/errors"

	"nvr/arp"
	"nvr/cuda"
)

type RTSP struct {
	Name             string
	Input            []string
	NetworkInterface string
	MacAddress       string
	Repeat           int

	// MulticastPort is the multicast port of the live stream.
	MulticastPort int `json:",omitempty"`
	// Multicast is the link to the live stream.
	Multicast string `json:",omitempty"`
	// Video is the link to the saved video.
	Video string `json:",omitempty"`
}

func (info RTSP) GetInput() ([]string, int, error) {
	var err error
	for i := 0; i < 10; i++ {
		var input []string
		input, err = info.getInput()
		if err == nil {
			return input, info.MulticastPort, nil
		}
		<-time.After(500 * time.Millisecond)
	}
	return nil, -1, err
}

func (info RTSP) getInput() ([]string, error) {
	if len(info.Input) == 0 {
		return nil, errors.Errorf("empty input")
	}
	last := info.Input[len(info.Input)-1]

	var input []string
	switch {
	case info.NetworkInterface != "":
		hws, err := arp.Scan(info.NetworkInterface)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		hw, ok := hws[info.MacAddress]
		if !ok {
			return nil, errors.Errorf("%#v", hws)
		}
		data := struct{ IP string }{IP: hw.IP}
		buf := bytes.NewBuffer(nil)
		if err := template.Must(template.New("").Parse(last)).Execute(buf, data); err != nil {
			return nil, errors.Errorf("%#v", data)
		}

		input = make([]string, len(info.Input))
		copy(input, info.Input[:len(info.Input)-1])
		input[len(input)-1] = string(buf.Bytes())
	default:
		input = info.Input
	}

	if len(input) == 1 {
		input = append([]string{"-i"}, input[0])
	}
	return input, nil
}

func (rtsp RTSP) Dir(recordDir string) string {
	return filepath.Join(recordDir, rtsp.Name)
}

func fixFFProbeArg(input []string) []string {
	fixed := make([]string, 0, len(input))
	i := 0
	for i < len(input) {
		inp := input[i]

		switch inp {
		case "-stream_loop":
			i += 2
		case "-re":
			i += 1
		default:
			fixed = append(fixed, inp)
			i++
		}
	}
	return fixed
}

func (rtsp RTSP) Prepare(recordDir string) error {
	input, _, err := rtsp.GetInput()
	if err != nil {
		return errors.Wrap(err, "")
	}
	if _, err := FFProbe(fixFFProbeArg(input)); err != nil {
		return errors.Wrap(err, "")
	}

	if err := os.MkdirAll(rtsp.Dir(recordDir), os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

type Track struct {
	Segment string
	Count   int
}

type Count struct {
	Src    string
	Config CountConfig

	Track Track

	TrackVideo string `json:",omitempty"`
}

func (c Count) Fill(recordDir string) Count {
	c.Config.Src = filepath.Join(recordDir, c.Src, IndexM3U8)
	c.Config.TrackIndex = filepath.Join(recordDir, c.Src+"Track", IndexM3U8)
	c.Config.TrackLog = filepath.Join(filepath.Dir(c.Config.TrackIndex), "track.json")
	return c
}

func (c Count) Prepare() error {
	if strings.HasPrefix(c.Config.AI.Device, "cuda") {
		if !cuda.IsAvailable() {
			return errors.Errorf("no cuda")
		}
	}

	if err := os.MkdirAll(filepath.Dir(c.Config.TrackIndex), os.ModePerm); err != nil {
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

func (c Count) LastTrack() (Track, error) {
	f, err := os.Open(c.Config.TrackLog)
	if err != nil {
		return Track{}, errors.Wrap(err, "")
	}
	defer f.Close()

	lines := make([]string, 0)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return Track{}, errors.Wrap(err, "")
	}

	if len(lines) == 0 {
		return Track{}, nil
	}

	last := lines[len(lines)-1]
	var track Track
	if err := json.Unmarshal([]byte(last), &track); err != nil {
		return Track{}, errors.Wrap(err, "")
	}
	return track, nil
}

func Update(db *sql.DB, table, id, columns string, values []interface{}) error {
	sqlStr := fmt.Sprintf(`UPDATE %s
	SET %s
	WHERE id=?`, table, columns)
	args := append(values, id)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := db.ExecContext(ctx, sqlStr, args...); err != nil {
		return errors.Wrap(err, fmt.Sprintf("\"%s\"", sqlStr))
	}
	return nil
}

type Record struct {
	ID    string
	RTSP  []RTSP
	Count []Count

	Err     string
	Create  time.Time
	Stop    time.Time
	Cleanup time.Time

	// Fields for display only.
	Link       string
	CreateTime string
	StopTime   string
}

func RecordDir(root, id string) string {
	yearStr := id[:4]
	dayStr := id[:8]
	dir := filepath.Join(root, yearStr, dayStr, id)
	return dir
}

func InsertRecord(db *sql.DB, r Record) error {
	rtspB, err := json.Marshal(r.RTSP)
	if err != nil {
		return errors.Wrap(err, "")
	}
	countB, err := json.Marshal(r.Count)
	if err != nil {
		return errors.Wrap(err, "")
	}

	sqlStr := "INSERT INTO " + TableRecord + `
	(id, rtsp, count, err, createAt, stop, cleanup) VALUES
	(?,  ?,    ?,     ?,  ?,         ?,    ?)`
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := db.ExecContext(ctx, sqlStr, r.ID, rtspB, countB, r.Err, r.Create, r.Stop, r.Cleanup); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func UpdateLastTrack(db *sql.DB, record Record, countID int) error {
	track, err := record.Count[countID].LastTrack()
	if err != nil {
		return errors.Wrap(err, "")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer tx.Rollback()

	// Read track.
	sqlStr := `SELECT count FROM ` + TableRecord + ` WHERE id=?`
	var countB []byte
	if err := tx.QueryRowContext(ctx, sqlStr, record.ID).Scan(&countB); err != nil {
		return errors.Wrap(err, "")
	}
	if err := json.Unmarshal(countB, &record.Count); err != nil {
		return errors.Wrap(err, "")
	}

	// Update track.
	if !(countID >= 0 && countID < len(record.Count)) {
		return errors.Errorf("%d %d %#v", countID, len(record.Count), record.Count)
	}
	record.Count[countID].Track = track

	// Save track.
	updatedB, err := json.Marshal(record.Count)
	if err != nil {
		return errors.Wrap(err, "")
	}
	sqlStr = `UPDATE ` + TableRecord + ` SET count=? WHERE id=?`
	if _, err := db.ExecContext(ctx, sqlStr, updatedB, record.ID); err != nil {
		return errors.Wrap(err, "")
	}

	// Commit.
	if err := tx.Commit(); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func GetRecord(db *sql.DB, id string) (Record, error) {
	records, err := SelectRecord(db, "WHERE id=?", []interface{}{id})
	if err != nil {
		return Record{}, errors.Wrap(err, "")
	}
	if len(records) == 0 {
		return Record{}, ErrNotFound
	}
	return records[0], nil
}

func SelectRecord(db *sql.DB, constraint string, args []interface{}) ([]Record, error) {
	sqlStr := `SELECT id, rtsp, count, err, createAt, stop, cleanup FROM ` + TableRecord + " " + constraint
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	rows, err := db.QueryContext(ctx, sqlStr, args...)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	defer rows.Close()

	records := make([]Record, 0)
	for rows.Next() {
		var r Record
		var rtspB, countB []byte
		if err := rows.Scan(&r.ID, &rtspB, &countB, &r.Err, &r.Create, &r.Stop, &r.Cleanup); err != nil {
			return nil, errors.Wrap(err, "")
		}
		if err := json.Unmarshal(rtspB, &r.RTSP); err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("%s", rtspB))
		}
		if err := json.Unmarshal(countB, &r.Count); err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("%s", countB))
		}
		records = append(records, r)
	}
	if err := rows.Err(); err != nil {
		return nil, errors.Wrap(err, "")
	}
	return records, nil
}
