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
)

var (
	serverDir = flag.String("d", "server", "server directory")
)

const (
	PathHLSIndex = "/HLSIndex"
	PathVideo    = "/Video"
	PathServe    = "/Serve"
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

type arpHW struct {
	ip string
	hw string
}

func arpScan(networkInterface string) (map[string]arpHW, error) {
	program := "arp-scan"
	arg := []string{
		"--interface=" + networkInterface,
		"-l",
		// Concise output to aid parsing.
		"-x",
	}
	cmd := exec.Command(program, arg...)
	b, err := cmd.CombinedOutput()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	hws := make(map[string]arpHW)
	scanner := bufio.NewScanner(bytes.NewBuffer(b))
	for scanner.Scan() {
		line := scanner.Text()
		cols := strings.Split(line, "\n")
		if len(cols) < 2 {
			return nil, errors.Wrap(err, fmt.Sprintf("%#v %s", cols, b))
		}

		var hw arpHW
		hw.ip = cols[0]
		hw.hw = cols[1]
		hws[hw.hw] = hw
	}
	if scanner.Err(); err != nil {
		return nil, errors.Wrap(err, "")
	}
	return hws, nil
}

type rtspInfo struct {
	// Fields related to the source stream.
	Name             string
	Link             string
	NetworkInterface string
	MacAddress       string
	Username         string
	Password         string
	Port             int
	Path             string

	// Fields related to background recording process.
	Pid   int
	Start time.Time

	// Fields related to UI.
	Live string
}

func (info rtspInfo) getLink() (string, error) {
	if info.Link != "" {
		return info.Link, nil
	}

	hws, err := arpScan(info.NetworkInterface)
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

type rtspMap struct {
	sync.RWMutex
	m map[string]rtspInfo
}

func newRTSPMap() *rtspMap {
	m := &rtspMap{}
	m.m = make(map[string]rtspInfo)
	return m
}

func (m *rtspMap) Set(info rtspInfo) {
	m.Lock()
	m.m[info.Name] = info
	m.Unlock()
}

func (m *rtspMap) All() []rtspInfo {
	m.RLock()
	all := make([]rtspInfo, 0, len(m.m))
	for _, info := range m.m {
		all = append(all, info)
	}
	m.RUnlock()
	return all
}

type countInfo struct {
	Config     nvr.CountConfig
	TrackVideo string

	Pid   int
	Start time.Time

	TrackLive string
}

type countMap struct {
	sync.RWMutex
	m map[string]countInfo
}

func newCountMap() *countMap {
	m := &countMap{}
	m.m = make(map[string]countInfo)
	return m
}

func (m *countMap) Set(info countInfo) {
	m.Lock()
	m.m[info.TrackVideo] = info
	m.Unlock()
}

func (m *countMap) All() []countInfo {
	m.RLock()
	all := make([]countInfo, 0, len(m.m))
	for _, info := range m.m {
		all = append(all, info)
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

	rtsp  *rtspMap
	count *countMap
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

	s.rtsp = newRTSPMap()
	s.count = newCountMap()

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

func (s *Server) startRTSP(recordDir string, info rtspInfo) (context.CancelFunc, chan error, error) {
	dir := filepath.Join(recordDir, info.Name)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return nil, nil, errors.Wrap(err, "")
	}
	getInput := func() (string, error) {
		return info.getLink()
	}
	onStart := func(pid int) {
		info.Pid = pid
		info.Start = time.Now()
		s.rtsp.Set(info)
	}
	recordFn := nvr.RecordVideoFn(dir, getInput, onStart)

	ctx, cancel := context.WithCancel(context.Background())
	errC := make(chan error)
	go func() {
		errC <- nvr.Loop(ctx, recordFn)
	}()
	return cancel, errC, nil
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
