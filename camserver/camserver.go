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
	TableJob     = "job"
	TableDeadJob = "deadjob"
	TableStat    = "stat"

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
			date TEXT,
			camera TEXT,
			n INTEGER,
			PRIMARY KEY (date, camera)
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
	Dir      string
	Filepath string
	Time     time.Time
}

func ProcessVideo(ctx context.Context, db *sql.DB, counter *Counter, arg ProcessVideoInput) error {
	dir := filepath.Join(arg.Dir, ProcessVideoDir, util.RunID())
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	mpegts := filepath.Join(dir, RawMPEGTSFilename)
	if err := toMPEGTS(ctx, mpegts, arg.Filepath); err != nil {
		return errors.Wrap(err, "")
	}

	countPath := filepath.Join(dir, CountVideoFilename)
	countOut, err := counter.Analyze(ctx, countPath, arg.Filepath)
	if err != nil {
		return errors.Wrap(err, "")
	}

	if err := incrStat(ctx, db, arg.Time, arg.Camera, countOut.Passed); err != nil {
		return errors.Wrap(err, "")
	}

	if err := os.WriteFile(filepath.Join(dir, util.DoneFilename), []byte{}, os.ModePerm); err != nil {
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

func incrStat(ctx context.Context, db *sql.DB, t time.Time, camera string, diff int) error {
	date := t.In(time.UTC).Format(util.FormatDate)

	tx, err := db.Begin()
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer tx.Rollback()

	selectStr := `SELECT n FROM ` + TableStat + ` WHERE date=? AND camera=?`
	var n int
	if err := tx.QueryRowContext(ctx, selectStr, date, camera).Scan(&n); err != nil {
		if err != sql.ErrNoRows {
			return errors.Wrap(err, "")
		}
	}

	insertStr := `REPLACE INTO ` + TableStat + ` (date, camera, n) VALUES (?, ?, ?)`
	if _, err := tx.ExecContext(ctx, insertStr, date, camera, n+diff); err != nil {
		return errors.Wrap(err, "")
	}

	if err := tx.Commit(); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func readStat(ctx context.Context, db *sql.DB, t time.Time, camera string) (int, error) {
	date := t.In(time.UTC).Format(util.FormatDate)
	selectStr := `SELECT n FROM ` + TableStat + ` WHERE date=? AND camera=?`
	var n int
	if err := db.QueryRowContext(ctx, selectStr, date, camera).Scan(&n); err != nil {
		return -1, errors.Wrap(err, "")
	}
	return n, nil
}
