package camserver

import (
	"camserver/ffmpeg"
	"camserver/util"
	"context"
	"database/sql"
	_ "embed"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/pkg/errors"
)

const (
	TableJob       = "job"
	TableDeadJob   = "deadjob"
	TableStat      = "stat"
	TableStatDedup = "stat_dedup"

	// Folder for the save raw video task.
	RawDir   = "raw"
	RawNoExt = "video"
	// Folder for the process video task.
	ProcessVideoDir    = "processvideo"
	RawMPEGTSFilename  = "raw.ts"
	CountVideoFilename = "count.ts"
)

func CreateTables(ctx context.Context, db *sql.DB) error {
	sqlStrs := []string{
		`CREATE TABLE ` + TableJob + ` (
			id TEXT,
			createAt INTEGER,
			job BLOB,
			duration INTEGER,
			shard INTEGER,
			lease INTEGER,
			retries INTEGER,
			PRIMARY KEY (id)
		) STRICT`,
		`CREATE TABLE ` + TableDeadJob + ` (
			id TEXT,
			createAt INTEGER,
			job BLOB,
			err TEXT,
			PRIMARY KEY (id)
		) STRICT`,
		`CREATE TABLE ` + TableStat + ` (
			dateHour TEXT,
			camera TEXT,
			n INTEGER,
			PRIMARY KEY (dateHour, camera)
		) STRICT`,
		`CREATE TABLE ` + TableStatDedup + ` (
			camera TEXT,
			video TEXT,
			t INTEGER,
			PRIMARY KEY (camera, video)
		) STRICT`,
	}
	for _, sqlStr := range sqlStrs {
		if _, err := db.ExecContext(ctx, sqlStr); err != nil {
			return errors.Wrap(err, fmt.Sprintf("\"%s\"", sqlStr))
		}
	}
	return nil
}

//go:embed count.py
var countPY string

//go:embed util.py
var utilPY string

type Scripts struct {
	Count string
	Util  string
}

func NewScripts(dir string) (Scripts, error) {
	s := Scripts{
		Count: filepath.Join(dir, "count.py"),
		Util:  filepath.Join(dir, "util.py"),
	}

	if err := os.MkdirAll(dir, 0775); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}

	if err := os.WriteFile(s.Count, []byte(countPY), 0644); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}
	if err := os.WriteFile(s.Util, []byte(utilPY), 0644); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}

	return s, nil
}

type ProcessVideoInput struct {
	Camera   string
	VideoID  string
	Dir      string
	Filepath string
	Time     time.Time
}

func ProcessVideo(ctx context.Context, db *sql.DB, scripts Scripts, countCfg CountConfig, arg ProcessVideoInput) error {
	dir := filepath.Join(arg.Dir, ProcessVideoDir, util.RunID())
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	mpegts := filepath.Join(dir, RawMPEGTSFilename)
	if err := toMPEGTS(ctx, mpegts, arg.Filepath); err != nil {
		return errors.Wrap(err, "")
	}

	countPath := filepath.Join(dir, CountVideoFilename)
	countOut, err := runCount(ctx, scripts.Count, countPath, arg.Filepath, countCfg)
	if err != nil {
		return errors.Wrap(err, "")
	}

	if err := incrStat(ctx, db, arg.Time, arg.Camera, arg.VideoID, countOut.Passed); err != nil {
		return errors.Wrap(err, "")
	}

	if err := util.WriteJSONFile(filepath.Join(dir, util.DoneFilename), countOut); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func toMPEGTS(ctx context.Context, dst, src string) error {
	const fps = 30
	teeMPEGTS := "[" + strings.Join([]string{
		"f=mpegts",
		"fifo_options=" + ffmpeg.TeeEscape([]string{
			"queue_size=" + strconv.Itoa(60*fps),
		}),
	}, ":") + "]" + dst
	arg := ffmpeg.Escape([]string{
		"-i", src,
		// No audio.
		"-an",
		// Variable frame rate, otherwise the HLS codec fails.
		// "-vsync", "vfr", // for old versions of ffmpeg
		"-fps_mode:v:0", "vfr",
		// Use H264 encoding.
		"-c:v", "libx264",
		// H264 high profile level 4.2 for maximum support across devices.
		"-profile:v", "high", "-level:v", "4.2",
		// Tee multiple outputs.
		"-f", "tee", "-map", "0:v", "-use_fifo", "1", teeMPEGTS,
	})
	cmd := exec.CommandContext(ctx, "ffmpeg", arg...)
	b, err := cmd.CombinedOutput()
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf("%s", b))
	}
	return nil
}

func incrStat(ctx context.Context, db *sql.DB, t time.Time, camera, videoID string, diff int) error {
	dateHour := t.In(time.UTC).Format(util.FormatDateHour)

	tx, err := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer tx.Rollback()

	selectVPStr := `SELECT t FROM ` + TableStatDedup + ` WHERE camera=? AND video=?`
	var prevT int
	err = tx.QueryRowContext(ctx, selectVPStr, camera, videoID).Scan(&prevT)
	switch err {
	case sql.ErrNoRows:
	case nil:
		return nil
	default:
		return errors.Wrap(err, "")
	}

	insertVPStr := `INSERT INTO ` + TableStatDedup + ` (camera, video, t) VALUES (?, ?, ?)`
	if _, err := tx.ExecContext(ctx, insertVPStr, camera, videoID, time.Now().Unix()); err != nil {
		return errors.Wrap(err, "")
	}

	selectStr := `SELECT n FROM ` + TableStat + ` WHERE dateHour=? AND camera=?`
	var n int
	if err := tx.QueryRowContext(ctx, selectStr, dateHour, camera).Scan(&n); err != nil {
		if err != sql.ErrNoRows {
			return errors.Wrap(err, "")
		}
	}

	insertStatStr := `REPLACE INTO ` + TableStat + ` (dateHour, camera, n) VALUES (?, ?, ?)`
	if _, err := tx.ExecContext(ctx, insertStatStr, dateHour, camera, n+diff); err != nil {
		return errors.Wrap(err, "")
	}

	if err := tx.Commit(); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func readStat(ctx context.Context, db *sql.DB, t time.Time, camera string) (int, error) {
	dateHour := t.In(time.UTC).Format(util.FormatDateHour)
	selectStr := `SELECT n FROM ` + TableStat + ` WHERE dateHour=? AND camera=?`
	var n int
	if err := db.QueryRowContext(ctx, selectStr, dateHour, camera).Scan(&n); err != nil {
		return -1, errors.Wrap(err, "")
	}
	return n, nil
}
