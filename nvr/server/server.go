package server

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/pkg/errors"

	"nvr"
)

const (
	PathStartRecord = "/StartRecord"
	PathStopRecord  = "/StopRecord"
	PathRecordPage  = "/RecordPage"
	PathHLSIndex    = "/HLSIndex"
	PathVideo       = "/Video"
	PathServe       = "/Serve"

	stdouterrFilename = "stdouterr.txt"
	statusFilename    = "status.txt"
)

func StartRecord(s *Server, w http.ResponseWriter, r *http.Request) (interface{}, error) {
	// id, err := startVideoFile(s, "sample/shilin20230826.mp4")
	id, err := startVideoWifi(s)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	resp := struct{ ID string }{ID: id}
	return resp, nil
}

func (s *Server) stopRecord(id string) error {
	rr, ok := s.records.get(id)
	if !ok {
		return errors.Errorf("not found")
	}
	rr.cancel()
	return nil
}

func StopRecord(s *Server, w http.ResponseWriter, r *http.Request) (interface{}, error) {
	id := r.FormValue("id")
	if err := s.stopRecord(id); err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("\"%s\"", id))
	}
	return struct{}{}, nil
}

func HLSIndex(s *Server, w http.ResponseWriter, r *http.Request) {
	fpath := r.FormValue("f")

	dir := filepath.Dir(fpath)

	f, err := os.Open(fpath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()
	out := bytes.NewBuffer(nil)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		var outLine string
		switch {
		case strings.HasPrefix(line, "#"):
			outLine = line
		default:
			outLine = s.serveURL(filepath.Join(dir, line))
		}
		out.WriteString(outLine + "\n")
	}
	if err := scanner.Err(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Write(out.Bytes())
}

//go:embed video.html
var videoHTML string
var videoTmpl = template.Must(template.New("").Parse(videoHTML))

func Video(s *Server, w http.ResponseWriter, r *http.Request) {
	src := r.FormValue("s")
	page := struct {
		Src string
	}{}
	page.Src = src
	videoTmpl.Execute(w, page)
}

func Serve(s *Server, w http.ResponseWriter, r *http.Request) {
	fpath := r.FormValue("f")
	http.ServeFile(w, r, fpath)
}

//go:embed record_page.html
var recordPageHTML string
var recordPageTmpl = template.Must(template.New("").Parse(recordPageHTML))

func RecordPage(s *Server, w http.ResponseWriter, r *http.Request) {
	id := r.FormValue("id")
	record, err := nvr.ReadRecord(s.RecordDir, id)
	if err != nil {
		http.Error(w, fmt.Sprintf("%+v", err), http.StatusInternalServerError)
		return
	}
	rd := nvr.RecordDir(s.RecordDir, record.ID)
	for i, rtsp := range record.RTSP {
		indexPath := filepath.Join(rd, rtsp.Name, nvr.IndexM3U8)
		record.RTSP[i].Video = s.videoURL(indexPath)
	}
	for i, c := range record.Count {
		record.Count[i].Track, err = c.LastTrack()
		if err != nil {
			http.Error(w, fmt.Sprintf("%+v", err), http.StatusInternalServerError)
			return
		}
		record.Count[i].TrackVideo = s.videoURL(c.Config.TrackIndex)
	}
	if err := recordPageTmpl.Execute(w, record); err != nil {
		log.Printf("%+v", err)
	}
}

//go:embed index.html
var indexHTML string
var indexTmpl = template.Must(template.New("").Parse(indexHTML))

func Index(s *Server, w http.ResponseWriter, r *http.Request) {
	page := struct {
		CurrentRecord nvr.Record
		StartURL      string
		StopURL       string

		LatestRecords []nvr.Record
	}{}

	records := s.records.all()
	for _, r := range records {
		if r.record.Stop.IsZero() {
			page.CurrentRecord = r.record
			page.CurrentRecord.Link = recordLink(r.record)
			break
		}
	}
	page.StartURL = PathStartRecord
	v := url.Values{}
	v.Set("id", page.CurrentRecord.ID)
	page.StopURL = PathStopRecord + "?" + v.Encode()

	var err error
	page.LatestRecords, err = nvr.ListRecord(s.RecordDir)
	if err != nil {
		http.Error(w, fmt.Sprintf("%+v", err), http.StatusInternalServerError)
		return
	}
	for i, r := range page.LatestRecords {
		page.LatestRecords[i].Link = recordLink(r)
		page.LatestRecords[i].CreateTime = r.Create.In(time.Local).Format(time.DateTime)
		page.LatestRecords[i].StopTime = r.Stop.In(time.Local).Format(time.DateTime)
	}

	if err := indexTmpl.Execute(w, page); err != nil {
		// log.Printf("%+v", err)
	}
}

type runningRecord struct {
	record nvr.Record
	cancel context.CancelFunc
}

type recordMap struct {
	sync.RWMutex
	m map[string]*runningRecord
}

func newRecordMap() *recordMap {
	m := &recordMap{}
	m.m = make(map[string]*runningRecord)
	return m
}

func (m *recordMap) set(rr *runningRecord) {
	m.Lock()
	defer m.Unlock()
	if _, ok := m.m[rr.record.ID]; ok {
		panic(fmt.Sprintf("%#v", m))
	}
	m.m[rr.record.ID] = rr
}

func (m *recordMap) del(id string) {
	m.Lock()
	delete(m.m, id)
	m.Unlock()
}

func (m *recordMap) get(id string) (*runningRecord, bool) {
	m.RLock()
	rr, ok := m.m[id]
	m.RUnlock()
	return rr, ok
}

func (m *recordMap) all() []*runningRecord {
	m.RLock()
	all := make([]*runningRecord, 0, len(m.m))
	for _, rr := range m.m {
		all = append(all, rr)
	}
	m.RUnlock()
	return all
}

type Server struct {
	ServeMux *http.ServeMux
	Server   http.Server

	ScriptDir string
	Scripts   nvr.Scripts

	RecordDir string

	records *recordMap
}

func NewServer(dir, addr string) (*Server, error) {
	s := &Server{}
	s.ServeMux = http.NewServeMux()
	s.Server.Addr = addr
	s.Server.Handler = s.ServeMux

	s.ScriptDir = filepath.Join(dir, "script")
	var err error
	s.Scripts, err = nvr.NewScripts(s.ScriptDir)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.RecordDir = filepath.Join(dir, "record")
	if err := os.MkdirAll(s.RecordDir, os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.records = newRecordMap()

	handleJSON(s, PathStartRecord, StartRecord)
	handleJSON(s, PathStopRecord, StopRecord)
	handleFunc(s, PathRecordPage, RecordPage)
	handleFunc(s, PathHLSIndex, HLSIndex)
	handleFunc(s, PathVideo, Video)
	handleFunc(s, PathServe, Serve)
	handleFunc(s, "/", Index)

	return s, nil
}

func (s *Server) serveURL(fpath string) string {
	v := url.Values{}
	v.Set("f", fpath)
	return PathServe + "?" + v.Encode()
}

func (s *Server) videoURL(indexPath string) string {
	v := url.Values{}
	v.Set("s", s.hlsIndex(indexPath))
	return PathVideo + "?" + v.Encode()
}

func (s *Server) hlsIndex(fpath string) string {
	v := url.Values{}
	v.Set("f", fpath)
	return PathHLSIndex + "?" + v.Encode()
}

func recordLink(record nvr.Record) string {
	v := url.Values{}
	v.Set("id", record.ID)
	return PathRecordPage + "?" + v.Encode()
}

func (s *Server) startRecord(record nvr.Record) (string, error) {
	now := time.Now()
	record.ID = nvr.TimeFormat(now)
	record.Create = now

	recordDir := nvr.RecordDir(s.RecordDir, record.ID)
	for i, c := range record.Count {
		record.Count[i] = c.Fill(recordDir)
	}
	for _, rtsp := range record.RTSP {
		if err := rtsp.Prepare(recordDir); err != nil {
			os.RemoveAll(recordDir)
			return "", errors.Wrap(err, fmt.Sprintf("%#v", rtsp))
		}
	}
	if err := nvr.WriteRecord(s.RecordDir, record); err != nil {
		return "", errors.Wrap(err, "")
	}

	recordsSet := make(chan struct{})
	go func() {
		rr := &runningRecord{record: record}
		err := s.startRunningRecord(rr, recordsSet)
		if err == nil {
			return
		}

		rr.record.Err = fmt.Sprintf("%+v", err)
		if err := nvr.WriteRecord(s.RecordDir, rr.record); err != nil {
			log.Printf("%+v", err)
		}
	}()
	<-recordsSet
	return record.ID, nil
}

func (s *Server) startRunningRecord(rr *runningRecord, recordsSet chan struct{}) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	rr.cancel = cancel
	s.records.set(rr)
	defer s.records.del(rr.record.ID)
	close(recordsSet)

	// Prepare RTSPs.
	recordDir := nvr.RecordDir(s.RecordDir, rr.record.ID)
	rtspInit := make(chan error, len(rr.record.RTSP))
	rtspDone := make(map[string]chan struct{}, len(rr.record.RTSP))
	for _, rtsp := range rr.record.RTSP {
		done := make(chan struct{})
		rtspDone[rtsp.Name] = done
		go func(rtsp nvr.RTSP) {
			defer close(done)

			dir := rtsp.Dir(recordDir)
			// Files must be in the same function scope as the loop.
			// This is to make sure that closing of files are always after the looping is done.
			// This prevents the loop from writing to closed files.
			stdouterrPath := filepath.Join(dir, stdouterrFilename)
			stdouterrF, err := os.Create(stdouterrPath)
			if err != nil {
				rtspInit <- errors.Wrap(err, "")
				return
			}
			defer stdouterrF.Close()
			statusPath := filepath.Join(dir, statusFilename)
			statusF, err := os.Create(statusPath)
			if err != nil {
				rtspInit <- errors.Wrap(err, "")
				return
			}
			defer statusF.Close()
			rtspInit <- nil

			fn := nvr.RecordVideoFn(dir, rtsp.GetLink, stdouterrF, stdouterrF, statusF)
			for {
				fn(ctx)
				select {
				case <-ctx.Done():
					return
				default:
				}
			}
		}(rtsp)
	}
	for i := 0; i < len(rr.record.RTSP); i++ {
		if err := <-rtspInit; err != nil {
			return errors.Wrap(err, "")
		}
	}

	// Prepare counts.
	countInit := make(chan error, len(rr.record.Count))
	countDone := make([]chan struct{}, 0, len(rr.record.Count))
	for _, c := range rr.record.Count {
		srcDoneC := rtspDone[c.Src]
		done := make(chan struct{})
		countDone = append(countDone, done)
		go func(c nvr.Count) {
			defer close(done)

			if err := c.Prepare(); err != nil {
				countInit <- errors.Wrap(err, "")
				return
			}
			// Files must be in the same function scope as the loop.
			// This is to make sure that closing of files are always after the looping is done.
			// This prevents the loop from writing to closed files.
			dir := filepath.Dir(c.Config.TrackIndex)
			stdouterrPath := filepath.Join(dir, stdouterrFilename)
			stdouterrF, err := os.Create(stdouterrPath)
			if err != nil {
				countInit <- errors.Wrap(err, "")
				return
			}
			defer stdouterrF.Close()
			statusPath := filepath.Join(dir, statusFilename)
			statusF, err := os.Create(statusPath)
			if err != nil {
				countInit <- errors.Wrap(err, "")
				return
			}
			defer statusF.Close()
			countInit <- nil

			fn := nvr.CountFn(s.Scripts.Count, c.Config, stdouterrF, stdouterrF, statusF)
			countCtx, countCancel := context.WithCancel(context.Background())
			go func() {
				defer countCancel()
				<-srcDoneC
				deadline := time.Now().Add(5 * time.Minute)
				for {
					// Stop if we have been lagging behind src for too long.
					if time.Now().After(deadline) {
						return
					}
					// Stop if we have finished processing.
					sameIndexErr := c.SameIndex()
					if sameIndexErr == nil {
						return
					}
					<-time.After(time.Second)
				}
			}()
			for {
				fn(countCtx)
				select {
				case <-countCtx.Done():
					return
				default:
				}
			}
		}(c)
	}
	for i := 0; i < len(rr.record.Count); i++ {
		if err := <-countInit; err != nil {
			return errors.Wrap(err, "")
		}
	}

	<-ctx.Done()
	rr.record.Stop = time.Now()
	if err := nvr.WriteRecord(s.RecordDir, rr.record); err != nil {
		return errors.Wrap(err, "")
	}

	for _, done := range countDone {
		<-done
	}
	rr.record.Cleanup = time.Now()
	if err := nvr.WriteRecord(s.RecordDir, rr.record); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func handleFunc(s *Server, httpPath string, fn func(*Server, http.ResponseWriter, *http.Request)) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		fn(s, w, r)
	}
	s.ServeMux.HandleFunc(httpPath, handler)
}

func handleJSON(s *Server, httpPath string, fn func(*Server, http.ResponseWriter, *http.Request) (interface{}, error)) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		res, err := fn(s, w, r)
		if err != nil {
			msg := struct {
				Error struct {
					Msg string
				}
			}{}
			msg.Error.Msg = fmt.Sprintf("%+v", err)
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(msg)
			return
		}
		if err := json.NewEncoder(w).Encode(res); err != nil {
			log.Printf("%+v", err)
			return
		}
	}
	s.ServeMux.HandleFunc(httpPath, handler)
}
