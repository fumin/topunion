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
	"os/exec"
	"path/filepath"
	"sort"
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
	PathStopRecord = "/StopRecord"
	PathRecordPage = "/RecordPage"
	PathHLSIndex = "/HLSIndex"
	PathVideo    = "/Video"
	PathServe    = "/Serve"

	valueFilename = "v.json"
	stdouterrFilename = "stdouterr.txt"
	statusFilename = "status.txt"
)

func StartRecord(s *Server, w http.ResponseWriter, r *http.Request) (interface{}, error) {
	id, err = s.startRecord(record)
	if err != nil {
		return errors.Wrap(err, "")
	}
	resp := struct{ID string}{ID: id}
	return resp, nil
}

func StopRecord(s *Server, w http.ResponseWriter, r *http.Request) error {
	id := r.FormValue("id")
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
		record.RTSP[i].Video = videoURL(indexPath)
	}
	for i, c := range record.Count {
		record.Count[i].TrackVideo = videoURL(c.Config.TrackIndex)
	}
	recordPageTmpl.Execute(w, record)
}

//go:embed index.html
var indexHTML string
var indexTmpl = template.Must(template.New("").Parse(indexHTML))

func Index(s *Server, w http.ResponseWriter, r *http.Request) {
	page := struct {
		CurrentRecord struct{
			ID string
			Link string
		}
	}{}

	records := s.records.all()
	for _, r := range records {
		if r.Stop.IsZero() {
			page.CurrentRecord.ID = r.ID
			v := url.Values{}
			v.Set("id", r.ID)
			page.CurrentRecord.Link = PathRecordPage + "?" + v.Encode()
			break
		}
	}

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
	info.Link = fmt.Sprintf("rtsp://%s:%s@%s:%d%s", info.Username, info.Password, hw.ip, info.Port, info.Path)

	return info.Link, nil
}

func (rtsp RTSP) prepare(recordDir string) error {
	dir := filepath.Join(recordDir, rtsp.Name)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

type Count struct {
	Src string
	Config     nvr.CountConfig

	TrackVideo string `json:",omitempty"`
}

func (c Count) fill(recordDir string) Count {
	c.Config.Src = filepath.Join(recordDir, count.Src, nvr.IndexM3U8)
	c.Config.TrackIndex = filepath.Join(recordDir, count.Src+"Track", nvr.IndexM3U8)
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

type Record struct {
	ID string
	RTSP []RTSP
	Count []Count

	Err string
	Create time.Time
	Stop time.Time
	Cleanup time.Time
}

func (r Record) Dir(root string) string {
	dayStr := record.ID[:8]
	dir := filepath.Join(root, dayStr, record.ID)
	return dir
}

type runningRecord struct {
	record Record
	cancel context.CancelFunc
}

type recordMap struct {
	sync.RWMutex
	m map[string]runningRecord
}

func newRecordMap() *recordMap {
	m := &recordMap{}
	m.m = make(map[string]runningRecord)
	return m
}

func (m *recordMap) set(rr runningRecord) {
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

func (m *recordMap) all() []runningRecord {
	m.RLock()
	all := make([]runningRecord, 0, len(m.m))
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

	recordDir := record.Dir(server.RecordDir)
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
	fpath := filepath.Join(server.RecordDir, dayStr, id, valueFilename)
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
	rr := &runningRecord{record: record}
	err := startRunningRecord(rr)
	if err == nil {
		return rr.record.ID, nil
	}

	rr.record.Err = fmt.Sprintf("%+v", err)
	if err := writeRecord(rr.record); err != nil {
		log.Printf("%+v", err)
	}
	return "", errors.Wrap(err, "")
}

func (s *Server) startRunningRecord(rr *runningRecord) error {
	now := time.Now()
	rr.record.ID = nvr.TimeFormat(now)
	rr.record.Create = now
	recordDir := rr.record.Dir(server.RecordDir)
	for i, c := range rr.record {
		rr.record.Count[i] = c.fill(recordDir)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	rr.cancel = cancel
	s.records.set(rr)
	defer s.records.del(rr.record.ID)
	if err := s.writeRecord(rr.record); err != nil {
		return errors.Wrap(err, "")
	}

	// Prepare RTSPs.
	fns := make([]func(context.Context))
	for _, rtsp := range record.RTSP {
		if err := rtsp.prepare(recordDir); err != nil {
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
		fn := rtspFn(dir, rtsp.getLink, stdouterrF, stdouterrF, statusF)
		fns = append(fns, fn)
	}
	// Prepare counts.
	for _, c := range rr.record.Count {
		if err := c.prepare(); err != nil {
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
		fn := nvr.CountFn(s.Scripts.Count, count.config, stdouterrF, stdouterrF, statusF)
		fns = append(fns, fn)
	}

	var wg sync.WaitGroup
	for _, fn := range fns {
		wg.Add(1)
		go func(){
			defer wg.Done()
			for {
				fn(ctx)
				select {
				case <-ctx.Done():
					return
				default:
				}
			}
		}()
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

	now := time.Now().In(time.UTC)
	dayStr := now.Format("20060102")
	recordName := nvr.TimeFormat(now)
	recordDir := filepath.Join(server.RecordDir, dayStr, recordName)
	recordDir = `/Users/shaoyu/a/topunion/nvr/server/record/20231010/20231010_091344_836093`

	rtsp0 := rtspInfo{
		Name:             "rtsp0",
		NetworkInterface: "",
		MacAddress:       "",
		Username:         "admin",
		Password:         "0000",
		Port:             8080,
		Path:             "/h264_ulaw.sdp",
	}
	// rtsp0.Link = "rtsp://localhost:8554/rtsp"
	rtsp0.Link = "sample/short.mp4"
	cancelRTSP0, rtsp0ErrC, err := server.startRTSP(recordDir, rtsp0)
	if err != nil {
		return errors.Wrap(err, "")
	}

	rtsp0Count := nvr.CountConfig{}
	rtsp0Count.Device = "cpu"
	rtsp0Count.Mask.Enable = false
	rtsp0Count.Yolo.Weights = "yolo_best.pt"
	rtsp0Count.Yolo.Size = 640
	rtsp0Count.Track.PrevCount = 10000
	cancelRTSP0Count, rtsp0CountErrC, err := server.startCount(recordDir, rtsp0.Name, rtsp0Count)
	if err != nil {
		return errors.Wrap(err, "")
	}

	log.Printf("listening at %s", server.Server.Addr)
	if err := server.Server.ListenAndServe(); err != http.ErrServerClosed {
		log.Printf("%+v", err)
	}

	cancelRTSP0()
	if err := <-rtsp0ErrC; err != nil {
		log.Printf("%+v", err)
	}
	cancelRTSP0Count()
	if err := <-rtsp0CountErrC; err != nil {
		log.Printf("%+v", err)
	}

	return nil
}
