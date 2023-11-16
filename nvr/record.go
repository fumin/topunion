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
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/pkg/errors"

	"nvr/arp"
	"nvr/ffmpeg"
	"nvr/util"
)

type Camera struct {
	Name             string
	Input            []string
	NetworkInterface string
	MacAddress       string
	Repeat           int

	// MPEGTS is the HTTP link to the MPEGTS stream.
	MPEGTS string `json:",omitempty"`
	// Video is the HTTP link to the saved video.
	Video string `json:",omitempty"`
}

func (cam Camera) GetInput() ([]string, error) {
	var err error
	for i := 0; ; i++ {
		var input []string
		input, err = cam.getInput()
		if err == nil {
			return input, nil
		}

		if i > 10 {
			break
		}
		<-time.After(500 * time.Millisecond)
	}
	return nil, err
}

func (cam Camera) getInput() ([]string, error) {
	if len(cam.Input) == 0 {
		return nil, errors.Errorf("empty input")
	}
	last := cam.Input[len(cam.Input)-1]

	var input []string
	switch {
	case cam.NetworkInterface != "":
		hws, err := arp.Scan(cam.NetworkInterface)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		hw, ok := hws[cam.MacAddress]
		if !ok {
			return nil, errors.Errorf("mac address \"%s\" not found in %#v", cam.MacAddress, hws)
		}
		data := struct{ IP string }{IP: hw.IP}
		buf := bytes.NewBuffer(nil)
		if err := template.Must(template.New("").Parse(last)).Execute(buf, data); err != nil {
			return nil, errors.Errorf("%#v", data)
		}

		input = make([]string, len(cam.Input))
		copy(input, cam.Input[:len(cam.Input)-1])
		input[len(input)-1] = string(buf.Bytes())
	default:
		input = cam.Input
	}

	if len(input) == 1 {
		input = append([]string{"-i"}, input[0])
	}
	return input, nil
}

func (cam Camera) Validate(ctx context.Context) error {
	input, err := cam.GetInput()
	if err != nil {
		return errors.Wrap(err, "")
	}
	fixed := ffmpeg.FixFFProbeInput(input)
	if _, err := ffmpeg.FFProbe(ctx, fixed); err != nil {
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

func (c Count) Validate(ctx context.Context, script string) error {
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer os.RemoveAll(dir)

	// Prepare src.
	srcDir := filepath.Join(dir, "src")
	if err := os.MkdirAll(srcDir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	cmd := strings.Split("ffmpeg -f lavfi -i smptebars -t 0.1 -f mpegts "+filepath.Join(srcDir, "0.ts"), " ")
	if b, err := exec.CommandContext(ctx, cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		return errors.Wrap(err, fmt.Sprintf("%s", b))
	}
	b := `#EXTM3U
#EXTINF:0.1,
0.ts
#EXT-X-ENDLIST`
	src := filepath.Join(srcDir, IndexM3U8)
	if err := os.WriteFile(src, []byte(b), os.ModePerm); err != nil {
		return errors.Wrap(err, fmt.Sprintf("\"%s\"", src))
	}

	// Prepare config.
	dstDir := filepath.Join(dir, "dst")
	if err := os.MkdirAll(dstDir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	dst := filepath.Join(dstDir, IndexM3U8)
	var config CountConfig
	config = c.Config
	config.Src = src
	config.TrackIndex = dst
	config.TrackLog = filepath.Join(filepath.Dir(dst), "track.json")
	cfg, err := json.Marshal(config)
	if err != nil {
		return errors.Wrap(err, "")
	}

	// Check that config works.
	stdouterr, err := exec.CommandContext(ctx, Python, script, "-c="+string(cfg)).CombinedOutput()
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf("\"%s\"", stdouterr))
	}

	return nil
}

func (c Count) Prepare(ctx context.Context) error {
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
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
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
	ID     string
	Camera []Camera
	Count  []Count

	Err     string
	Create  time.Time
	Stop    time.Time
	Cleanup time.Time

	// Fields for display only.
	CreateTime string
	Eggs       int
	StopTime   string
	Link       string
}

func RecordDir(root, id string) string {
	yearStr := id[:4]
	dayStr := id[:8]
	dir := filepath.Join(root, yearStr, dayStr, id)
	return dir
}

func InsertRecord(db *sql.DB, root string, create time.Time, r Record) (Record, error) {
	r.ID = util.TimeFormat(create)
	r.Create = create

	recordDir := RecordDir(root, r.ID)
	for i, c := range r.Count {
		r.Count[i] = c.Fill(recordDir)
	}

	cameraB, err := json.Marshal(r.Camera)
	if err != nil {
		return Record{}, errors.Wrap(err, "")
	}
	countB, err := json.Marshal(r.Count)
	if err != nil {
		return Record{}, errors.Wrap(err, "")
	}

	sqlStr := "INSERT INTO " + TableRecord + `
	(id, camera, count, err, createAt, stop, cleanup) VALUES
	(?,  ?,    ?,     ?,  ?,         ?,    ?)`
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := db.ExecContext(ctx, sqlStr, r.ID, cameraB, countB, r.Err, r.Create, r.Stop, r.Cleanup); err != nil {
		return Record{}, errors.Wrap(err, "")
	}
	return r, nil
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
	// Allocate memory instead of reusing record.Count to avoid data races.
	// This is because we are updating counts with the new track, while other goroutines read record.Count.
	var counts []Count
	if err := json.Unmarshal(countB, &counts); err != nil {
		return errors.Wrap(err, "")
	}

	// Update track.
	if !(countID >= 0 && countID < len(counts)) {
		return errors.Errorf("%d %d %#v", countID, len(counts), counts)
	}
	counts[countID].Track = track

	// Save track.
	updatedB, err := json.Marshal(counts)
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
		return Record{}, util.ErrNotFound
	}
	return records[0], nil
}

func SelectRecord(db *sql.DB, constraint string, args []interface{}) ([]Record, error) {
	sqlStr := `SELECT id, camera, count, err, createAt, stop, cleanup FROM ` + TableRecord + " " + constraint
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
		var cameraB, countB []byte
		if err := rows.Scan(&r.ID, &cameraB, &countB, &r.Err, &r.Create, &r.Stop, &r.Cleanup); err != nil {
			return nil, errors.Wrap(err, "")
		}
		if err := json.Unmarshal(cameraB, &r.Camera); err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("%s", cameraB))
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
