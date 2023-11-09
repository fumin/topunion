package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/pkg/errors"

	"nvr"
	"nvr/server"
)

var (
	serverDir = flag.String("d", "devData", "server directory")
)

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	s, err := server.NewServer(*serverDir, "")
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer s.Close()

	if err := nvr.CreateTables(s.DB); err != nil {
		return errors.Wrap(err, "")
	}
	if err := insertRecords(s); err != nil {
		return errors.Wrap(err, "")
	}

	return nil
}

func insertRecords(s *server.Server) error {
	now := time.Now()
	monthFirst := time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, now.Location())

	records := make([]nvr.Record, 0)
	for t := monthFirst; t.Before(now); t = t.Add(24 * time.Hour) {
		create := t.Add(time.Duration(rand.Intn(8*60)) * time.Minute)
		r := nvr.Record{Create: create}

		for i := 0; i < 2; i++ {
			cam := nvr.Camera{Name: fmt.Sprintf("camera%d", i)}
			r.Camera = append(r.Camera, cam)
			count := nvr.Count{Src: cam.Name, Track: nvr.Track{Count: 8000 + rand.Intn(4000)}}
			r.Count = append(r.Count, count)
		}
		r.Stop = r.Create.Add(time.Duration(rand.Intn(8*60)) * time.Minute)
		r.Cleanup = r.Stop.Add(time.Duration(rand.Intn(2*60)) * time.Second)

		records = append(records, r)
	}

	for _, r := range records {
		if _, err := nvr.InsertRecord(s.DB, s.RecordDir, r.Create, r); err != nil {
			return errors.Wrap(err, fmt.Sprintf("%#v", r))
		}
	}

	return nil
}
