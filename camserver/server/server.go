package server

import (
	"cmp"
	"context"
	"database/sql"
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"

	"camserver"
	"camserver/util"
)

const (
	DBFilename = "db.sqlite"

	PathLive  = "/Live"
	PathServe = "/Serve"
)

type CameraStatus struct {
	ID   string
	T    time.Time
	Live string

	Err string
}

func getCameraStatus(s *Server, id string) (CameraStatus, error) {
	camDir := filepath.Join(s.VideoDir, id)
	day, err := latestDay(camDir)
	if err != nil {
		return CameraStatus{}, errors.Wrap(err, "")
	}
	dayDir := filepath.Join(camDir, day[0], day[1])

	var t *time.Time
	segs, err := os.ReadDir(dayDir)
	if err != nil {
		return CameraStatus{}, errors.Wrap(err, "")
	}
	for i := len(segs) - 1; i >= 0; i-- {
		segDE := segs[i]

		rawDir := filepath.Join(dayDir, segDE.Name(), camserver.RawDir)
		doneRaw, err := util.GetDoneTry(rawDir)
		if err != nil {
			continue
		}
		doneBody := struct{ T time.Time }{}
		if err := util.ReadJSONFile(filepath.Join(rawDir, doneRaw, util.DoneFilename), &doneBody); err != nil {
			continue
		}
		t = &doneBody.T
	}
	if t == nil {
		return CameraStatus{}, errors.Errorf("not found %#v", day)
	}
	status := CameraStatus{
		ID:   id,
		T:    *t,
		Live: PathLive + "?" + url.Values{"c": {id}}.Encode(),
	}
	return status, nil
}

//go:embed tmpl/status.html
var statusHTML string
var statusTmpl = template.Must(template.New("").Parse(statusHTML))

func Status(s *Server, w http.ResponseWriter, r *http.Request) {
	camIDs := make([]string, 0, len(s.Camera))
	for id := range s.Camera {
		camIDs = append(camIDs, id)
	}
	slices.Sort(camIDs)

	type StatusPage struct {
		Camera []CameraStatus
	}
	var page StatusPage
	for _, id := range camIDs {
		c, err := getCameraStatus(s, id)
		if err != nil {
			c.ID = id
			c.Err = fmt.Sprintf("%+v", err)
		}
		page.Camera = append(page.Camera, c)
	}

	if err := statusTmpl.Execute(w, page); err != nil {
		log.Printf("%+v", err)
	}
}

func Live(s *Server, w http.ResponseWriter, r *http.Request) {
	cam := r.FormValue("c")

	camDir := filepath.Join(s.VideoDir, cam)
	day, err := latestDay(camDir)
	if err != nil {
		http.Error(w, fmt.Sprintf("%+v", err), http.StatusBadRequest)
		return
	}
	dayDir := filepath.Join(camDir, day[0], day[1])
	playlist, err := getPlaylist(dayDir)
	if err != nil {
		http.Error(w, fmt.Sprintf("%+v", err), http.StatusBadRequest)
		return
	}

	// Rewrite file path URLs.
	for i, s := range playlist.Segment {
		v := url.Values{}
		v.Set("f", s.URL)
		playlist.Segment[i].URL = PathServe + "?" + v.Encode()
	}

	w.Write(playlist.Bytes())
}

func UploadVideo(s *Server, w http.ResponseWriter, r *http.Request) (interface{}, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := r.ParseMultipartForm(32 * 1024 * 1024); err != nil {
		return nil, errors.Wrap(err, "")
	}

	// Camera ID.
	cam := r.FormValue("c")
	if cam == "" {
		return nil, errors.Errorf("empty camera ID")
	}
	if _, ok := s.Camera[cam]; !ok {
		return nil, errors.Errorf("unknown camera")
	}

	// Uploaded file.
	f, fh, err := r.FormFile("f")
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	const formatDatetime = "20060102_150405"
	if len(fh.Filename) < len(formatDatetime) {
		return nil, errors.Errorf("%d", len(fh.Filename))
	}
	t, err := time.ParseInLocation(formatDatetime, fh.Filename[:len(formatDatetime)], time.UTC)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	vidDur, err := util.ReadVideoDuration(ctx, f)
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return -1, errors.Wrap(err, "")
	}

	// Save uploaded file.
	ext := filepath.Ext(fh.Filename)
	videoID := strings.TrimSuffix(fh.Filename, ext)
	videoDir := filepath.Join(s.VideoDir, cam, t.Format("2006"), t.Format(util.FormatDate), videoID)
	reqDir := filepath.Join(videoDir, camserver.RawDir, util.RunID())
	if err := os.MkdirAll(reqDir, os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}
	dst := filepath.Join(reqDir, camserver.RawNoExt+ext)
	if err := util.WriteFile(dst, f); err != nil {
		return nil, errors.Wrap(err, "")
	}

	// Send background job.
	jobArg := camserver.ProcessVideoInput{
		Camera:   cam,
		Dir:      videoDir,
		Filepath: dst,
		Time:     t,
	}
	job := Job{Func: JobProcessVideo, Arg: jobArg}
	if err := SendJob(ctx, s.DB, job); err != nil {
		return nil, errors.Wrap(err, "")
	}

	doneBody := struct {
		T        time.Time
		Duration float64
	}{
		T:        t,
		Duration: vidDur,
	}
	doneB, err := json.Marshal(doneBody)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	if err := os.WriteFile(filepath.Join(reqDir, util.DoneFilename), doneB, os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}

	resp := struct {
		T int64
	}{}
	resp.T = s.StartTime.Unix()
	return resp, nil
}

type stat struct {
	dateHour string
	camera   string
	n        int
}

func readLatestStat(ctx context.Context, db *sql.DB) ([]stat, error) {
	sqlStr := `SELECT dateHour, camera, n FROM ` + camserver.TableStat + ` ORDER BY dateHour DESC LIMIT 80`
	rows, err := db.QueryContext(ctx, sqlStr)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	defer rows.Close()

	stats := make([]stat, 0)
	for rows.Next() {
		var s stat
		if err := rows.Scan(&s.dateHour, &s.camera, &s.n); err != nil {
			return nil, errors.Wrap(err, "")
		}
		stats = append(stats, s)
	}
	if err := rows.Err(); err != nil {
		return nil, errors.Wrap(err, "")
	}
	return stats, nil
}

//go:embed tmpl/index.html
var indexHTML string
var indexTmpl = template.Must(template.New("").Parse(indexHTML))

func Index(s *Server, w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	stats, err := readLatestStat(ctx, s.DB)
	if err != nil {
		http.Error(w, fmt.Sprintf("%+v", err), http.StatusBadRequest)
		return
	}

	type datum struct {
		DateHour time.Time
		Camera   map[string]int
	}
	dataM := make(map[string]datum)
	camerasM := make(map[string]struct{})
	for _, s := range stats {
		d, ok := dataM[s.dateHour]
		if !ok {
			d.Camera = make(map[string]int)
			d.DateHour, err = time.ParseInLocation(util.FormatDateHour, s.dateHour, time.UTC)
			if err != nil {
				http.Error(w, fmt.Sprintf("%+v", err), http.StatusBadRequest)
				return
			}
			d.DateHour = d.DateHour.In(util.TaipeiTZ)
		}
		d.Camera[s.camera] = s.n

		dataM[s.dateHour] = d
		camerasM[s.camera] = struct{}{}
	}
	data := make([]datum, 0, len(dataM))
	for _, d := range dataM {
		data = append(data, d)
	}
	slices.SortFunc(data, func(a, b datum) int { return -cmp.Compare(a.DateHour.Unix(), b.DateHour.Unix()) })
	cameras := make([]string, 0, len(camerasM))
	for c := range camerasM {
		cameras = append(cameras, c)
	}
	slices.Sort(cameras)

	page := struct {
		Camera []string
		Data   []datum
	}{}
	page.Camera = cameras
	page.Data = data
	if err := indexTmpl.Execute(w, page); err != nil {
		log.Printf("%+v", err)
	}
}

func Serve(s *Server, w http.ResponseWriter, r *http.Request) {
	fpath := r.FormValue("f")
	http.ServeFile(w, r, fpath)
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

type CameraConfig struct {
	ID     string
	Height int
	Width  int
	Count  camserver.CountConfig
}

type Camera struct {
	Config  CameraConfig
	Counter *camserver.Counter
}

type Config struct {
	Name string
	// Directory to store data.
	Dir string
	// Address to listen to.
	Addr string

	Camera []CameraConfig

	// Suppport setting database max connections to alleviate sqlite issue:
	// https://github.com/mattn/go-sqlite3/issues/209
	SqliteMaxConn int
}

type Server struct {
	C         Config
	StartTime time.Time

	ServeMux *http.ServeMux
	Server   http.Server

	ScriptDir string
	Scripts   camserver.Scripts

	BackgroundProcessDir string
	DB                   *sql.DB
	VideoDir             string

	Camera map[string]Camera
}

//go:embed static
var staticFS embed.FS

func NewServer(config Config) (*Server, error) {
	s := &Server{C: config, StartTime: time.Now()}
	s.ServeMux = http.NewServeMux()
	s.Server.Addr = s.C.Addr
	s.Server.Handler = s.ServeMux

	s.ScriptDir = filepath.Join(s.C.Dir, "script")
	var err error
	s.Scripts, err = camserver.NewScripts(s.ScriptDir)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.BackgroundProcessDir = filepath.Join(s.C.Dir, "proc")
	if err := os.MkdirAll(s.BackgroundProcessDir, os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}

	dbPath := filepath.Join(s.C.Dir, DBFilename)
	dbV := url.Values{}
	dbV.Set("_journal_mode", "WAL")
	// https://github.com/mattn/go-sqlite3/issues/209
	dbV.Set("_busy_timeout", "5000")
	s.DB, err = sql.Open("sqlite3", "file:"+dbPath+"?"+dbV.Encode())
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	if s.C.SqliteMaxConn != 0 {
		s.DB.SetMaxOpenConns(s.C.SqliteMaxConn)
	}

	s.VideoDir = filepath.Join(s.C.Dir, "video")
	if err := os.MkdirAll(s.VideoDir, os.ModePerm); err != nil {
		return nil, errors.Wrap(err, "")
	}

	s.Camera = make(map[string]Camera)
	for _, camCfg := range s.C.Camera {
		if err := util.IsAlphaNumeric(camCfg.ID); err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("\"%s\" %#v", camCfg.ID, camCfg))
		}
		if _, ok := s.Camera[camCfg.ID]; ok {
			return nil, errors.Errorf("duplicate camera \"%s\" %#v", camCfg.ID, camCfg)
		}
		cam := Camera{Config: camCfg}

		camCfg.Count.Height = camCfg.Height
		camCfg.Count.Width = camCfg.Width
		cam.Counter, err = camserver.NewCounter(s.Scripts.Count, camCfg.Count)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		s.Camera[cam.Config.ID] = cam
	}

	handleFunc(s, "/Status", Status)
	handleFunc(s, PathLive, Live)
	handleJSON(s, "/UploadVideo", UploadVideo)
	handleFunc(s, PathServe, Serve)
	s.ServeMux.Handle("/static/", http.FileServer(http.FS(staticFS)))
	handleFunc(s, "/", Index)

	return s, nil
}

func (s *Server) Close() error {
	if err := s.DB.Close(); err != nil {
		return errors.Wrap(err, "")
	}
	return nil
}
