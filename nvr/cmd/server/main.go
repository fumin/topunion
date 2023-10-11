package main

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"flag"
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
	"nvr/arp"
)

var (
	serverDir = flag.String("d", "server", "server directory")
)

const (
	PathStartRecord = "/StartRecord"
	PathStopRecord  = "/StopRecord"
	PathRecordPage  = "/RecordPage"
	PathHLSIndex    = "/HLSIndex"
	PathVideo       = "/Video"
	PathServe       = "/Serve"

	valueFilename     = "v.json"
	stdouterrFilename = "stdouterr.txt"
	statusFilename    = "status.txt"
)

func StartRecord(s *Server, w http.ResponseWriter, r *http.Request) (interface{}, error) {
	var record Record
	rtsp0 := RTSP{
		Name:             "rtsp0",
		Link:             "sample/shilin20230826.mp4",
		NetworkInterface: "",
		MacAddress:       "",
		Username:         "admin",
		Password:         "0000",
		Port:             8080,
		Path:             "/h264_ulaw.sdp",
	}
	record.RTSP = append(record.RTSP, rtsp0)

	count0 := Count{Src: rtsp0.Name}
	count0.Config.Device = "cuda:0"
	count0.Config.Mask.Enable = false
	count0.Config.Yolo.Weights = "yolo_best.pt"
	count0.Config.Yolo.Size = 640
	count0.Config.Track.PrevCount = 10000
	record.Count = append(record.Count, count0)

	id, err := s.startRecord(record)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	resp := struct{ ID string }{ID: id}
	return resp, nil
}

func StopRecord(s *Server, w http.ResponseWriter, r *http.Request) (interface{}, error) {
	id := r.FormValue("id")
	rr, ok := s.records.get(id)
	if !ok {
		return nil, errors.Errorf("not found")
	}
	rr.cancel()
	s.records.del(id)
	resp := struct{}{}
	return resp, nil
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
	record, err := s.readRecord(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	rd := record.Dir(s.RecordDir)
	for i, rtsp := range record.RTSP {
		indexPath := filepath.Join(rd, rtsp.Name, nvr.IndexM3U8)
		record.RTSP[i].Video = s.videoURL(indexPath)
	}
	for i, c := range record.Count {
		record.Count[i].TrackVideo = s.videoURL(c.Config.TrackIndex)
	}
	recordPageTmpl.Execute(w, record)
}

//go:embed index.html
var indexHTML string
var indexTmpl = template.Must(template.New("").Parse(indexHTML))

func Index(s *Server, w http.ResponseWriter, r *http.Request) {
	page := struct {
		CurrentRecord struct {
			ID   string
			Link string
		}
		StartURL string
		StopURL string
	}{}

	records := s.records.all()
	for _, r := range records {
		if r.record.Stop.IsZero() {
			page.CurrentRecord.ID = r.record.ID
			v := url.Values{}
			v.Set("id", r.record.ID)
			page.CurrentRecord.Link = PathRecordPage + "?" + v.Encode()
			break
		}
	}
	page.StartURL = PathStartRecord
	v := url.Values{}
	v.Set("id", page.CurrentRecord.ID)
	page.StopURL = PathStopRecord + "?" + v.Encode()

	indexTmpl.Execute(w, page)
}

type RTSP struct {
	Name             string
	Link             string
	NetworkInterface string
	MacAddress       string
	Username         string
	Password         string
	Port             int
	Path             string

	Video string `json:",omitempty"`
}

func (info RTSP) getLink() (string, error) {
	if info.Link != "" {
		return info.Link, nil
	}

	hws, err := arp.Scan(info.NetworkInterface)
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	hw, ok := hws[info.MacAddress]
	if !ok {
		return "", errors.Errorf("%#v %#v", info, hws)
	}
	info.Link = fmt.Sprintf("rtsp://%s:%s@%s:%d%s", info.Username, info.Password, hw.IP, info.Port, info.Path)

	return info.Link, nil
}

func (rtsp RTSP) prepare(recordDir string) (string, error) {
	dir := filepath.Join(recordDir, rtsp.Name)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return "", errors.Wrap(err, "")
	}
	return dir, nil
}

type Count struct {
	Src    string
	Config nvr.CountConfig

	TrackVideo string `json:",omitempty"`
}

func (c Count) fill(recordDir string) Count {
	c.Config.Src = filepath.Join(recordDir, c.Src, nvr.IndexM3U8)
	c.Config.TrackIndex = filepath.Join(recordDir, c.Src+"Track", nvr.IndexM3U8)
	return c
}

func (c Count) prepare() error {
	trackDir := filepath.Dir(c.Config.TrackIndex)
	if err := os.MkdirAll(trackDir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	// Wait for src to appear.
	var err error
	for i := 0; i < nvr.HLSTime*3/2; i++ {
		_, err = os.Stat(c.Config.Src)
		if err == nil {
			break
		}
		<-time.After(time.Second)
	}
	if err != nil {
		return errors.Wrap(err, "")
	}

	return nil
}

func (c Count) sameIndex() (bool, error) {
	srcN, err := fileNumLines(c.Config.Src)
	if err != nil {
		return false, errors.Wrap(err, "")
	}
	trackN, err := fileNumLines(c.Config.TrackIndex)
	if err != nil {
		return false, errors.Wrap(err, "")
	}
	return srcN == trackN, nil
}

func fileNumLines(fpath string) (int, error) {
	f, err := os.Open(fpath)
	if err != nil {
		return -1, errors.Wrap(err, "")
	}
	defer f.Close()

	lines := 0
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lines++
	}
	if err := scanner.Err(); err != nil {
		return -1, errors.Wrap(err, "")
	}
	return lines, nil
}

type Record struct {
	ID    string
	RTSP  []RTSP
	Count []Count

	Err     string
	Create  time.Time
	Stop    time.Time
	Cleanup time.Time
}

func (r Record) Dir(root string) string {
	dayStr := r.ID[:8]
	dir := filepath.Join(root, dayStr, r.ID)
	return dir
}

type runningRecord struct {
	record Record
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

func (s *Server) writeRecord(record Record) error {
	b, err := json.Marshal(record)
	if err != nil {
		return errors.Wrap(err, "")
	}

	recordDir := record.Dir(s.RecordDir)
	if err := os.MkdirAll(recordDir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}

	fpath := filepath.Join(recordDir, valueFilename)
	if err := os.WriteFile(fpath, b, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (s *Server) readRecord(id string) (Record, error) {
	if len(id) < 8 {
		return Record{}, errors.Errorf("%d", len(id))
	}
	dayStr := id[:8]
	fpath := filepath.Join(s.RecordDir, dayStr, id, valueFilename)
	b, err := os.ReadFile(fpath)
	if err != nil {
		return Record{}, errors.Wrap(err, "")
	}
	var record Record
	if err := json.Unmarshal(b, &record); err != nil {
		return Record{}, errors.Wrap(err, "")
	}
	return record, nil
}

func (s *Server) startRecord(record Record) (string, error) {
	now := time.Now()
	record.ID = nvr.TimeFormat(now)
	record.Create = now
	recordDir := record.Dir(s.RecordDir)
	for i, c := range record.Count {
		record.Count[i] = c.fill(recordDir)
	}

	if err := s.writeRecord(record); err != nil {
		return "", errors.Wrap(err, "")
	}

	recordsSet := make(chan struct{})
	go func(){
		rr := &runningRecord{record: record}
		err := s.startRunningRecord(rr, recordsSet)
		if err == nil {
			return
		}

		rr.record.Err = fmt.Sprintf("%+v", err)
		if err := s.writeRecord(rr.record); err != nil {
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
	rtspFns := make([]func(context.Context), 0, len(rr.record.RTSP))
	recordDir := rr.record.Dir(s.RecordDir)
	for _, rtsp := range rr.record.RTSP {
		dir, err := rtsp.prepare(recordDir)
		if err != nil {
			return errors.Wrap(err, "")
		}
		stdouterrPath := filepath.Join(dir, stdouterrFilename)
		stdouterrF, err := os.Create(stdouterrPath)
		if err != nil {
			return errors.Wrap(err, "")
		}
		defer stdouterrF.Close()
		statusPath := filepath.Join(dir, statusFilename)
		statusF, err := os.Create(statusPath)
		if err != nil {
			return errors.Wrap(err, "")
		}
		defer statusF.Close()
		fn := nvr.RecordVideoFn(dir, rtsp.getLink, stdouterrF, stdouterrF, statusF)
		rtspFns = append(rtspFns, fn)
	}
	var wg sync.WaitGroup
	for _, fn := range rtspFns {
		wg.Add(1)
		go func(fn func(context.Context)) {
			defer wg.Done()
			for {
				fn(ctx)
				select {
				case <-ctx.Done():
					return
				default:
				}
			}
		}(fn)
	}

	// Prepare counts.
	type countFn struct{
		count Count
		fn func(context.Context)
	}
	countFns := make([]countFn, 0, len(rr.record.Count))
	for _, c := range rr.record.Count {
		if err := c.prepare(); err != nil {
			return errors.Wrap(err, "")
		}
		dir := filepath.Dir(c.Config.TrackIndex)
		stdouterrPath := filepath.Join(dir, stdouterrFilename)
		stdouterrF, err := os.Create(stdouterrPath)
		if err != nil {
			return errors.Wrap(err, "")
		}
		defer stdouterrF.Close()
		statusPath := filepath.Join(dir, statusFilename)
		statusF, err := os.Create(statusPath)
		if err != nil {
			return errors.Wrap(err, "")
		}
		defer statusF.Close()
		fn := nvr.CountFn(s.Scripts.Count, c.Config, stdouterrF, stdouterrF, statusF)
		countFns = append(countFns, countFn{count: c, fn: fn})
	}
	for _, cfn := range countFns {
		wg.Add(1)
		go func(cfn countFn) {
			defer wg.Done()
			for {
				cfn.fn(context.Background())
				ok, err := cfn.count.sameIndex()
				if err == nil && ok {
					return
				}
			}
		}(cfn)
	}

	<-ctx.Done()
	rr.record.Stop = time.Now()
	s.records.set(rr)
	if err := s.writeRecord(rr.record); err != nil {
		return errors.Wrap(err, "")
	}

	wg.Wait()
	rr.record.Cleanup = time.Now()
	if err := s.writeRecord(rr.record); err != nil {
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

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lmicroseconds | log.Llongfile)

	if err := mainWithErr(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func mainWithErr() error {
	addr := ":8080"
	server, err := NewServer(*serverDir, addr)
	if err != nil {
		return errors.Wrap(err, "")
	}

	log.Printf("listening at %s", server.Server.Addr)
	if err := server.Server.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("%+v", err)
	}
	return nil
}
