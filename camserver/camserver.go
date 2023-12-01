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

	"github.com/pkg/errors"
)

const (
	TableJob  = "job"
	TableStat = "stat"

	JobDir             = "job"
	RawMPEGTSFilename  = "raw.ts"
	CountVideoFilename = "count.ts"
)

func CreateTables(ctx context.Context, db *sql.DB) error {
	sqlStrs := []string{
		`CREATE TABLE ` + TableJob + ` (
			id TEXT,
			job BLOB,
			lease INTEGER,
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
	Filepath string
}

func ProcessVideo(ctx context.Context, db *sql.DB, arg ProcessVideoInput) error {
	runID := util.RandID()
	dir := filepath.Join(filepath.Dir(arg.Filepath), JobDir, runID)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	mpegts := filepath.Join(dir, RawMPEGTSFilename)
	if err := toMPEGTS(ctx, mpegts, arg.Filepath); err != nil {
		return errors.Wrap(err, "")
	}
	countPath := filepath.Join(dir, CountVideoFilename)
	if err := countEgg(ctx, countPath, arg.Filepath); err != nil {
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

func countEgg(ctx context.Context, dst, src string) error {
	return nil
}
