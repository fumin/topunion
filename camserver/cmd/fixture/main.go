package main

import (
	"context"
	"database/sql"
	"flag"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"

	"camserver"
	"camserver/server"
)

var (
	dir = flag.String("d", "devData", "data directory")
)

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

	return nil
}
