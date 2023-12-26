package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"

	"camserver"
	"camserver/server"
	"camserver/server/config"
	"camserver/util"
)

var (
	dir = flag.String("d", "devData", "data directory")
)

func insertStat(ctx context.Context, db *sql.DB) error {
	type statistic struct {
		t      time.Time
		camera string
		n      int
	}
	stats := make([]statistic, 0)
	start := time.Date(2007, 1, 2, 5, 0, 0, 0, util.TaipeiTZ)
	for i := 0; i < 256; i++ {
		t := start.Add(time.Duration(i) * time.Hour)
		stats = append(stats, statistic{t: t, camera: "camera0", n: 1100 + i})
		stats = append(stats, statistic{t: t, camera: "camera1", n: 1000 + i})
	}

	for _, s := range stats {
		sqlStr := `INSERT INTO ` + camserver.TableStat + ` (dateHour, camera, n) VALUES (?, ?, ?)`
		dateHour := s.t.In(time.UTC).Format(util.FormatDateHour)
		if _, err := db.ExecContext(ctx, sqlStr, dateHour, s.camera, s.n); err != nil {
			return errors.Wrap(err, fmt.Sprintf("%#v", s))
		}
	}
	return nil
}

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	if err := os.MkdirAll(*dir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	if err := os.WriteFile(filepath.Join(*dir, "config.json"), config.Dev, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	dbPath := filepath.Join(*dir, server.DBFilename)
	dbV := url.Values{}
	dbV.Set("_journal_mode", "WAL")
	db, err := sql.Open("sqlite3", "file:"+dbPath+"?"+dbV.Encode())
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := camserver.CreateTables(ctx, db); err != nil {
		return errors.Wrap(err, "")
	}

	if err := insertStat(ctx, db); err != nil {
		return errors.Wrap(err, "")
	}

	return nil
}
