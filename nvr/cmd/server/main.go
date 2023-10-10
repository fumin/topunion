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
	"nvr/util"
)

var (
	serverDir = flag.String("d", "server", "server directory")
)

const (
	PathHLSIndex = "/HLSIndex"
	PathVideo    = "/Video"
	PathServe    = "/Serve"

	valueFilename = "v.json"
)

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

func Quit(s *Server, w http.ResponseWriter, r *http.Request) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if err := s.Server.Shutdown(ctx); err != nil {
			log.Printf("%+v", err)
		}
	}()
}

//go:embed index.html
var indexHTML string
var indexTmpl = template.Must(template.New("").Parse(indexHTML))

func Index(s *Server, w http.ResponseWriter, r *http.Request) {
	rtsps := s.rtsp.All()
	now := time.Now()
	for i, info := range rtsps {
		rtsps[i].Live = s.videoURL(now, info.Name)
	}
	sort.Slice(rtsps, func(i, j int) bool { return rtsps[i].Name < rtsps[j].Name })

	counts := s.count.All()
	for i, info := range counts {
		counts[i].TrackLive = s.videoURL(now, info.TrackVideo)
	}
	sort.Slice(counts, func(i, j int) bool { return counts[i].TrackVideo < counts[j].TrackVideo })

	page := struct {
		RTSP  []rtspInfo
		Count []countInfo
	}{}
	page.RTSP = rtsps
	page.Count = counts
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

	hws, err := util.ARPScan(info.NetworkInterface)
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

type Count struct {
	Config     nvr.CountConfig

	TrackVideo string `json:",omitempty"`
}

type Record struct {
	ID string
	RTSP []RTSP
	Count []Count

	Create time.Time
	Quit time.Time
	Cleanup time.Time
}

type cancelDone struct {
	cancel context.CancelFunc
	done chan struct{}
}

type runningRecord struct {
	record Record

	rtspCancels []*cancelDone
	countCancels []*cancelDone
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

func (m *recordMap) Set(rr runningRecord) {
	m.Lock()
	m.m[rr.record.ID] = rr
	m.Unlock()
}

func (m *recordMap) All() []runningRecord {
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

	handleFunc(s, PathHLSIndex, HLSIndex)
	handleFunc(s, PathVideo, Video)
	handleFunc(s, PathServe, Serve)
	handleFunc(s, "/Quit", Quit)
	handleFunc(s, "/", Index)

	return s, nil
}

func (s *Server) serveURL(fpath string) string {
	v := url.Values{}
	v.Set("f", fpath)
	return PathServe + "?" + v.Encode()
}

func (s *Server) videoURL(anyT time.Time, name string) string {
	t := anyT.In(time.UTC)
	indexPath, err := s.videoIndex(t, name)
	if err != nil {
		return ""
	}
	v := url.Values{}
	v.Set("s", s.hlsIndex(indexPath))
	return PathVideo + "?" + v.Encode()
}

func (s *Server) hlsIndex(fpath string) string {
	v := url.Values{}
	v.Set("f", fpath)
	return PathHLSIndex + "?" + v.Encode()
}

func (s *Server) videoIndex(qt time.Time, name string) (string, error) {
	t := qt.In(time.UTC)
	dayDir := filepath.Join(s.RecordDir, t.Format("20060102"))
	entries, err := os.ReadDir(dayDir)
	if err != nil {
		return "", errors.Wrap(err, "")
	}

	lastRecord := ""
	for i := len(entries) - 1; i >= 0; i-- {
		rec := entries[i].Name()
		startT, err := nvr.TimeParse(rec)
		if err != nil {
			return "", errors.Wrap(err, fmt.Sprintf("\"%s\"", rec))
		}
		if startT.Before(t) {
			lastRecord = rec
			break
		}
	}
	if lastRecord == "" {
		return "", errors.Errorf("not found %s", dayDir)
	}

	indexFName := filepath.Join(dayDir, lastRecord, name, "index.m3u8")
	return indexFName, nil
}

func (s *Server) writeRecord(record Record) error {
	b, err := json.Marshal(record)
	if err != nil {
		return errors.Wrap(err, "")
	}

	dayStr := record.ID[:8]
	recordDir := filepath.Join(server.RecordDir, dayStr, record.ID)
	fpath := filepath.Join(recordDir, valueFilename)

	if err := os.WriteFile(fpath, b, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (s *Server) startRecord(rtsp []RTSP, count []Count) error {
	record := runningRecord{
		record: Record{
			ID: nvr.TimeFormat(time.Now()),
			RTSP: rtsp,
			Count: count,
			Create: now,
		}
	}

	dayStr := record.record.ID[:8]
	recordDir := filepath.Join(server.RecordDir, dayStr, record.record.ID)
	// recordDir = `/Users/shaoyu/a/topunion/nvr/server/record/20231010/20231010_091344_836093`
	if err := os.MkdirAll(recordDir, os.ModePerm); err != nil {
		return errors.Wrap(err, "")
	}
	if err := s.writeRecord(record.record); err != nil {
		return errors.Wrap(err, "")
	}

}

func (s *Server) startRTSP(recordDir string, info rtspInfo, pidFile, stdout, stderr *os.File) (*cancelDone, error) {
	dir := filepath.Join(recordDir, info.Name)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}
	onStart := func(pid int) {
		ss := []string{
			nvr.TimeFormat(time.Now()),
			strconv.Itoa(pid),
			"start",
		}
		pidFile.Write([]byte(strings.Join(ss, ",")+"\n"))
	}
	recordFn := nvr.RecordVideoFn(dir, info.getLink, stdout, stderr, onStart)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct)
	cd := &cancelDone{cancel: cancel, done: done}
	go func() {
		err := recordFn(ctx)
		ss := []string{
			nvr.TimeFormat(time.Now()),
		}

		errC <- nvr.Loop(ctx, recordFn)
	}()
	return cd, nil
}

func (s *Server) startCount(recordDir, src string, config nvr.CountConfig) (context.CancelFunc, chan error, error) {
	config.Src = filepath.Join(recordDir, src, nvr.IndexM3U8)
	// Wait for src to appear.
	var err error
	for i := 0; i < nvr.HLSTime*3/2; i++ {
		_, err := os.Stat(config.Src)
		if err == nil {
			break
		}
		<-time.After(time.Second)
	}
	if err != nil {
		return nil, nil, errors.Wrap(err, "")
	}

	trackVideo := src + "Track"
	trackDir := filepath.Join(recordDir, trackVideo)
	if err := os.MkdirAll(trackDir, os.ModePerm); err != nil {
		return nil, nil, errors.Wrap(err, "")
	}
	config.TrackIndex = filepath.Join(trackDir, nvr.IndexM3U8)

	onStart := func(pid int) {
		info := countInfo{Config: config, TrackVideo: trackVideo}
		info.Pid = pid
		info.Start = time.Now()
		s.count.Set(info)
	}
	loopFn := nvr.CountFn(s.Scripts.Count, config, onStart)

	ctx, cancel := context.WithCancel(context.Background())
	errC := make(chan error)
	go func() {
		errC <- nvr.Loop(ctx, loopFn)
	}()
	return cancel, errC, nil
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
