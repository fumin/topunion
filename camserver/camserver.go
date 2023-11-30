package camserver

import (
	"context"
	"database/sql"
	_ "embed"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
)

const (
	TableJob  = "job"
	TableStat = "stat"
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
	log.Printf("process video %s", arg.Filepath)
	return nil
}
