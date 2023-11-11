package server

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"database/sql"
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"

	"nvr"
	"nvr/util"
)

const (
	PathStartRecord = "/StartRecord"
	PathStopRecord  = "/StopRecord"
	PathGetRecord   = "/GetRecord"

	PathRecordPage  = "/RecordPage"
	PathControl     = "/Control"
	PathMPEGTSServe = "/MPEGTSServe"
	PathMPEGTS      = "/MPEGTS"
	PathHLSIndex    = "/HLSIndex"
	PathVideo       = "/Video"
	PathServe       = "/Serve"

	ErrorLogFilename = "error.txt"
)

func StartRecord(s *Server, w http.ResponseWriter, r *http.Request) (interface{}, error) {
	id, err := startSMPTE(s)
	// id, err := startVideoFile(s, "sample/shilin20230826.mp4")
	// id, err := startVideoWifi(s)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	resp := struct{ ID string }{ID: id}
	return resp, nil
}

func StopRecord(s *Server, w http.ResponseWriter, r *http.Request) (interface{}, error) {
	id := r.FormValue("id")
	if err := s.records.cancel(id); err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("\"%s\"", id))
	}
	return struct{}{}, nil
}

func GetRecord(s *Server, w http.ResponseWriter, r *http.Request) (interface{}, error) {
	id := r.FormValue("id")
	record, err := nvr.GetRecord(s.DB, id)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf("\"%s\"", id))
	}
	record = s.displayRecord(record, PathMPEGTS)

	// for i := range record.Count {
	// 	record.Count[i].Track.Count = rand.Intn(100)
	// }

	return record, nil
}

//go:embed tmpl/control.html
var controlHTML string
var controlTmpl = template.Must(template.New("").Parse(controlHTML))

func Control(s *Server, w http.ResponseWriter, r *http.Request) {
	page := struct {
		Record   nvr.Record
		StartURL string
		StopURL  string
		GetURL   string
	}{}

	records := s.records.running()
	if len(records) > 0 {
		page.Record = s.displayRecord(records[0], PathMPEGTSServe)
	}

	page.StartURL = PathStartRecord
	v := url.Values{}
	v.Set("id", page.Record.ID)
	page.StopURL = PathStopRecord + "?" + v.Encode()
	page.GetURL = PathGetRecord + "?" + v.Encode()

	controlTmpl.Execute(w, page)
}

//go:embed tmpl/mpegts.html
var mpegtsHTML string
var mpegtsTmpl = template.Must(template.New("").Parse(mpegtsHTML))

func MPEGTS(s *Server, w http.ResponseWriter, r *http.Request) {
	page := struct {
		URL string
	}{}

	v := url.Values{}
	v.Set("a", r.FormValue("a"))
	page.URL = PathMPEGTSServe + "?" + v.Encode()

	mpegtsTmpl.Execute(w, page)
}

func MPEGTSServe(s *Server, w http.ResponseWriter, r *http.Request) {
	addrStr := r.FormValue("a")
	addr, err := net.ResolveUDPAddr("udp", addrStr)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	ifi, err := net.InterfaceByName("lo0")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/octet-stream")

	conn, err := net.ListenMulticastUDP("udp", ifi, addr)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer conn.Close()

	// Cannot use io.CopyBuffer, since it does not use our provided buffer.
	// This is because conn implements WriterTo, which io.CopyBuffer directly uses.
	// The buffer size VLCUDPLen should match the source or else there will be conn.Read errors.
	b := make([]byte, util.VLCUDPLen)
	for {
		if err := conn.SetReadDeadline(time.Now().Add(15 * time.Second)); err != nil {
			break
		}
		n, err := conn.Read(b)
		if err != nil {
			break
		}
		if _, err := w.Write(b[:n]); err != nil {
			break
		}
	}
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

//go:embed tmpl/video.html
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

//go:embed tmpl/record_page.html
var recordPageHTML string
var recordPageTmpl = template.Must(template.New("").Parse(recordPageHTML))

func RecordPage(s *Server, w http.ResponseWriter, r *http.Request) {
	id := r.FormValue("id")
	record, err := nvr.GetRecord(s.DB, id)
	if err != nil {
		http.Error(w, fmt.Sprintf("%+v", err), http.StatusInternalServerError)
		return
	}
	record = s.displayRecord(record, PathMPEGTS)
	if err := recordPageTmpl.Execute(w, record); err != nil {
		log.Printf("%+v", err)
	}
}

//go:embed tmpl/index.html
var indexHTML string
var indexTmpl = template.Must(template.New("").Parse(indexHTML))

func Index(s *Server, w http.ResponseWriter, r *http.Request) {
	now := time.Now()
	month1st := time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, now.Location())
	lastMonth1st := month1st.AddDate(0, -1, 0)
	records, err := nvr.SelectRecord(s.DB, "WHERE id >= ?", []interface{}{util.TimeFormat(lastMonth1st)})
	if err != nil {
		http.Error(w, fmt.Sprintf("%+v", err), http.StatusInternalServerError)
		return
	}
	slices.SortFunc(records, func(a, b nvr.Record) int {
		return -cmp.Compare(a.Create.UnixNano(), b.Create.UnixNano())
	})

	mon := now.AddDate(0, 0, -int(now.Weekday()-time.Monday))
	thisMonday := time.Date(mon.Year(), mon.Month(), mon.Day(), 0, 0, 0, 0, now.Location())
	lastMonday := thisMonday.AddDate(0, 0, -7)
	thisWeek, lastWeek, thisMonth, lastMonth := 0, 0, 0, 0
	for _, r := range records {
		if r.Create.After(thisMonday) {
			thisWeek += r.Count[len(r.Count)-1].Track.Count
		}
		if r.Create.After(lastMonday) && r.Create.Before(thisMonday) {
			lastWeek += r.Count[len(r.Count)-1].Track.Count
		}
		if r.Create.After(month1st) {
			thisMonth += r.Count[len(r.Count)-1].Track.Count
		}
		if r.Create.After(lastMonth1st) && r.Create.Before(month1st) {
			lastMonth += r.Count[len(r.Count)-1].Track.Count
		}
	}

	latests := records
	if len(latests) > 30 {
		latests = latests[:30]
	}
	for i, r := range latests {
		latests[i] = s.displayRecord(r, PathMPEGTS)
	}

	page := struct {
		ThisWeek  int
		LastWeek  int
		ThisMonth int
		LastMonth int

		Records []nvr.Record
	}{}
	page.ThisWeek = thisWeek
	page.LastWeek = lastWeek
	page.ThisMonth = thisMonth
	page.LastMonth = lastMonth
	page.Records = latests

	if err := indexTmpl.Execute(w, page); err != nil {
		// log.Printf("%+v", err)
	}
}

type runningRecord struct {
	record   nvr.Record
	cancel   context.CancelFunc
	canceled bool
	ips      []string
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

func (m *recordMap) set(ipM *ipMap, id string, rr *runningRecord) error {
	m.Lock()
	defer m.Unlock()

	// Assign rtsp streams with unused IPs.
	var err error
	rr.ips, err = ipM.get(len(rr.record.Camera))
	if err != nil {
		return errors.Wrap(err, "")
	}

	if _, ok := m.m[id]; ok {
		return errors.Errorf("duplicate id %s %#v", id, m.m)
	}
	m.m[rr.record.ID] = rr
	return nil
}

func (m *recordMap) cancel(id string) error {
	m.Lock()
	defer m.Unlock()

	rr, ok := m.m[id]
	if !ok {
		return errors.Errorf("not found")
	}
	rr.cancel()
	rr.canceled = true
	return nil
}

func (m *recordMap) del(ipM *ipMap, id string) {
	m.Lock()
	defer m.Unlock()

	rr, ok := m.m[id]
	if !ok {
		return
	}

	ipM.putBack(rr.ips)
	delete(m.m, id)
}

func (m *recordMap) ips(id string) []string {
	m.RLock()
	defer m.RUnlock()

	rr, ok := m.m[id]
	if !ok {
		return nil
	}

	ips := make([]string, len(rr.ips))
	copy(ips, rr.ips)
	return ips
}

func (m *recordMap) running() []nvr.Record {
	m.RLock()
	defer m.RUnlock()

	running := make([]nvr.Record, 0, len(m.m))
	for _, rr := range m.m {
		if rr.canceled {
			continue
		}
		running = append(running, rr.record)
	}
	return running
}

type ipMap struct {
	net *net.IPNet

	sync.RWMutex
	m map[string]struct{}
}

func newIPMap(cidr string) (*ipMap, error) {
	_, ipnet, err := net.ParseCIDR(cidr)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	m := &ipMap{net: ipnet, m: make(map[string]struct{})}
	return m, nil
}

func (m *ipMap) get(n int) ([]string, error) {
	m.Lock()
	defer m.Unlock()

	broadcast := util.BroadcastAddr(m.net)

	unused := make([]string, 0, n)
	// Allocate the iterating ip, instead of using m.ip directly.
	// This is because the iteration will modify ip.
	ip := net.IP(make([]byte, len(m.net.IP)))
	copy(ip, m.net.IP)
	// Skip the first address, as it is the subnet address.
	util.Inc(ip)
	for ; m.net.Contains(ip); util.Inc(ip) {
		if ip.Equal(broadcast) {
			continue
		}

		if len(unused) >= n {
			break
		}

		s := ip.String()
		if _, ok := m.m[s]; ok {
			continue
		}

		unused = append(unused, s)
	}
	if len(unused) != n {
		return nil, errors.Errorf("insufficient")
	}

	for _, s := range unused {
		m.m[s] = struct{}{}
	}

	return unused, nil
}

func (m *ipMap) putBack(ips []string) {
	m.Lock()
	defer m.Unlock()

	for _, ip := range ips {
		delete(m.m, ip)
	}
}

type Server struct {
	ServeMux *http.ServeMux
	Server   http.Server

	ScriptDir string
	Scripts   nvr.Scripts

	DB        *sql.DB
	RecordDir string

	ips     *ipMap
	records *recordMap
}

//go:embed static
var staticFS embed.FS

func NewServer(dir, addr, multicast string) (*Server, error) {
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

	dbPath := filepath.Join(dir, "db.sqlite")
	dbV := url.Values{}
	dbV.Set("_journal_mode", "WAL")
	s.DB, err = sql.Open("sqlite3", "file:"+dbPath+"?"+dbV.Encode())
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.RecordDir = filepath.Join(dir, "record")
	if err := os.MkdirAll(s.RecordDir, os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.ips, err = newIPMap(multicast)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	s.records = newRecordMap()

	handleJSON(s, PathStartRecord, StartRecord)
	handleJSON(s, PathStopRecord, StopRecord)
	handleJSON(s, PathGetRecord, GetRecord)

	handleFunc(s, PathRecordPage, RecordPage)
	handleFunc(s, PathControl, Control)
	handleFunc(s, PathMPEGTSServe, MPEGTSServe)
	handleFunc(s, PathMPEGTS, MPEGTS)
	handleFunc(s, PathHLSIndex, HLSIndex)
	handleFunc(s, PathVideo, Video)
	handleFunc(s, PathServe, Serve)
	s.ServeMux.Handle("/static/", http.FileServer(http.FS(staticFS)))
	handleFunc(s, "/", Index)

	return s, nil
}

func (s *Server) Close() error {
	s.cancelRecordings()

	err := s.DB.Close()
	if err != nil {
		return errors.Wrap(err, "")
	}
	return nil
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
	// Validate fields.
	camNames := make(map[string]struct{}, len(record.Camera))
	for _, cam := range record.Camera {
		if _, ok := camNames[cam.Name]; ok {
			return "", errors.Errorf("duplicate name %s %#v", cam.Name, camNames)
		}
		camNames[cam.Name] = struct{}{}

		if err := cam.Validate(); err != nil {
			return "", errors.Wrap(err, fmt.Sprintf("%#v", cam))
		}
	}

	// Write to database.
	var err error
	record, err = nvr.InsertRecord(s.DB, s.RecordDir, time.Now(), record)
	if err != nil {
		return "", errors.Wrap(err, "")
	}

	// Start recording in background.
	recordsSet := make(chan struct{})
	go func() {
		rr := &runningRecord{record: record}
		err := s.startRunningRecord(rr, recordsSet)
		if err == nil {
			return
		}

		rr.record.Err = fmt.Sprintf("%+v", err)
		if err := nvr.Update(s.DB, nvr.TableRecord, rr.record.ID, "err=?", []interface{}{rr.record.Err}); err != nil {
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
	if err := s.records.set(s.ips, rr.record.ID, rr); err != nil {
		return errors.Wrap(err, "")
	}
	defer s.records.del(s.ips, rr.record.ID)
	close(recordsSet)

	// Prepare cameras.
	recordDir := nvr.RecordDir(s.RecordDir, rr.record.ID)
	cameraDone := make(map[string]chan struct{}, len(rr.record.Camera))
	for i, cam := range rr.record.Camera {
		multicast := ffmpegMulticast(rr.ips[i])

		dir := filepath.Join(recordDir, cam.Name)
		if err := os.MkdirAll(dir, os.ModePerm); err != nil {
			return errors.Wrap(err, "")
		}

		done := make(chan struct{})
		cameraDone[cam.Name] = done
		go func(cam nvr.Camera, multicast string) {
			defer close(done)

			repeat := cam.Repeat
			if repeat == 0 {
				repeat = math.MaxInt
			}
			getInput := func() ([]string, string, error) {
				input, err := cam.GetInput()
				if err != nil {
					return nil, "", errors.Wrap(err, "")
				}
				return input, multicast, nil
			}
			fn := nvr.RecordVideoFn(dir, getInput)
			for i := 0; i < repeat; i++ {
				fn(ctx)
				select {
				case <-ctx.Done():
					return
				default:
				}
			}
		}(cam, multicast)
	}
	allCameraDone := make(chan struct{})
	go func() {
		defer close(allCameraDone)
		for _, c := range cameraDone {
			<-c
		}
	}()

	// Prepare counts.
	countDone := make([]chan struct{}, 0, len(rr.record.Count))
	for i, c := range rr.record.Count {
		if err := c.Prepare(); err != nil {
			return errors.Wrap(err, "")
		}

		srcDoneC := cameraDone[c.Src]
		done := make(chan struct{})
		countDone = append(countDone, done)
		go func(countID int, c nvr.Count) {
			defer close(done)

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
			dir := filepath.Dir(c.Config.TrackIndex)
			processDone := make(chan struct{})
			go func() {
				defer close(processDone)
				fn := nvr.CountFn(dir, s.Scripts.Count, c.Config)
				for {
					fn(countCtx)
					select {
					case <-countCtx.Done():
						return
					default:
					}
				}
			}()
			errLog := util.NewErrorLogger(filepath.Join(dir, ErrorLogFilename))
			defer errLog.Close()
		Loop:
			for {
				errLog.E(func() error { return nvr.UpdateLastTrack(s.DB, rr.record, countID) })
				select {
				case <-processDone:
					break Loop
				case <-time.After(nvr.HLSTime * time.Second):
				}
			}
			errLog.E(func() error { return nvr.UpdateLastTrack(s.DB, rr.record, countID) })
		}(i, c)
	}

	select {
	case <-ctx.Done():
	case <-allCameraDone:
	}
	if err := nvr.Update(s.DB, nvr.TableRecord, rr.record.ID, "stop=?", []interface{}{time.Now()}); err != nil {
		return errors.Wrap(err, "")
	}

	for _, done := range countDone {
		<-done
	}
	if err := nvr.Update(s.DB, nvr.TableRecord, rr.record.ID, "cleanup=?", []interface{}{time.Now()}); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}

func (s *Server) cancelRecordings() {
	// Only hold the lock for a brief time.
	// This is to avoid a deadlock when the background recording wants to delete itself from s.records.
	s.records.Lock()
	for _, rr := range s.records.m {
		rr.cancel()
	}
	s.records.Unlock()

	for i := 1; ; i++ {
		s.records.RLock()
		ids := make([]string, 0, len(s.records.m))
		for id := range s.records.m {
			ids = append(ids, id)
		}
		s.records.RUnlock()
		if len(ids) == 0 {
			break
		}

		if i%10 == 0 {
			log.Printf("waiting for records to cleanup %#v", ids)
		}
		<-time.After(1 * time.Second)
	}
}

func (s *Server) displayRecord(r nvr.Record, mpegtsPath string) nvr.Record {
	// Record fields.
	r.CreateTime = r.Create.In(time.Local).Format(time.DateTime)
	if len(r.Count) > 0 {
		r.Eggs = r.Count[len(r.Count)-1].Track.Count
	}
	r.StopTime = r.Stop.In(time.Local).Format(time.DateTime)
	r.Link = recordLink(r)

	// Camera fields.
	var ips []string
	if r.Stop.IsZero() {
		ips = s.records.ips(r.ID)
	}
	rDir := nvr.RecordDir(s.RecordDir, r.ID)
	for i, cam := range r.Camera {
		indexPath := filepath.Join(rDir, cam.Name, nvr.IndexM3U8)
		r.Camera[i].Video = s.videoURL(indexPath)

		if len(ips) > 0 {
			v := url.Values{}
			v.Set("a", ffmpegMulticast(ips[i]))
			r.Camera[i].MPEGTS = mpegtsPath + "?" + v.Encode()
		}
	}

	// Count fields.
	for i, c := range r.Count {
		r.Count[i].TrackVideo = s.videoURL(c.Config.TrackIndex)
	}

	return r
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

func ffmpegMulticast(ip string) string {
	return ip + ":10000"
}
