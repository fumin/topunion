package nvr

import (
	"context"
	"database/sql"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/pkg/errors"

	"nvr/ffmpeg"
	"nvr/util"
)

const (
	IndexM3U8 = "index.m3u8"
	HLSTime   = 10

	Python = "python3"

	TableRecord = "record"
)

//go:embed count.py
var countPY string

type Scripts struct {
	Count string
}

func NewScripts(dir string) (Scripts, error) {
	s := Scripts{
		Count: filepath.Join(dir, "count.py"),
	}

	if err := os.MkdirAll(dir, 0775); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}

	if err := os.WriteFile(s.Count, []byte(countPY), 0644); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}

	return s, nil
}

type CountConfig struct {
	TrackIndex string
	TrackLog   string
	Src        string
	AI         struct {
		Smart  bool
		Device string
		Mask   struct {
			Enable bool
			Crop   struct {
				X int
				Y int
				W int
			}
			Mask struct {
				Slope float64
				Y     int
				H     int
			}
		}
		Yolo struct {
			Weights string
			Size    int
		}
		Track struct {
		}
	}
}

func CountFn(dir, script string, cfg CountConfig) func(context.Context) {
	run := func(ctx context.Context, stdout, stderr *os.File) (*exec.Cmd, error) {
		c, err := json.Marshal(cfg)
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		arg := []string{script, "-c=" + string(c)}

		cmd := exec.CommandContext(ctx, Python, arg...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		cmd.Stdout = stdout
		cmd.Stderr = stderr
		cmd.Cancel = func() error {
			_, err := stdin.Write([]byte("q\n"))
			return err
		}
		cmd.WaitDelay = 60 * time.Second

		return cmd, nil
	}
	return util.NewCmdFn(dir, run)
}

func RecordVideoFn(dir string, getInput func() ([]string, string, error)) func(context.Context) {
	const program = "ffmpeg"
	const fps = 30
	const hlsFlags = "" +
		// Append to the same HLS index file.
		"append_list" +
		// Add segment index in filename to deduplicate filenames within the same second.
		"+second_level_segment_index" +
		// Add program time info explicitly.
		// If we do not, with append_list turned on, ffmpeg will nonetheless insist on adding garbage program time, resulting in a worse situation.
		"+program_date_time"
	run := func(ctx context.Context, stdout, stderr *os.File) (*exec.Cmd, error) {
		input, multicastAddr, err := getInput()
		if err != nil {
			return nil, errors.Wrap(err, "")
		}

		segmentFName := filepath.Join(dir, "%s_%%06d.ts")
		if runtime.GOOS == "windows" {
			// strftime does not support "%s" in windows.
			segmentFName = filepath.Join(dir, "%Y%m%d_%H%M%S_%%06d.ts")
		}
		indexFName := filepath.Join(dir, IndexM3U8)
		teeHLS := "[" + strings.Join([]string{
			"f=hls",
			"fifo_options=" + ffmpeg.TeeEscape([]string{
				"queue_size=" + strconv.Itoa(60*fps),
			}),
			// 10 seconds per segment.
			"hls_time=" + strconv.Itoa(HLSTime),
			// No limit on number of segments.
			"hls_list_size=0",
			// Use strftime syntax for segment filenames.
			"strftime=1",
			"hls_flags=" + hlsFlags,
			"hls_segment_filename=" + ffmpeg.TeeEscape([]string{segmentFName}),
		}, ":") + "]" + indexFName
		udpOutput := fmt.Sprintf("udp://%s/?pkt_size=%d", multicastAddr, util.VLCUDPLen)

		teeUDP := "[" + strings.Join([]string{
			"f=mpegts",
			"fifo_options=" + ffmpeg.TeeEscape([]string{
				// "queue_size=" + strconv.Itoa(60*fps),
				// The below options are not supported ffmpeg 4.4.2.
				"drop_pkts_on_overflow=1",
				// "attempt_recovery=1",
				// "recover_any_error=1",
			}),
		}, ":") + "]" + udpOutput
		arg := ffmpeg.Escape(append(input, []string{
			// No audio.
			"-an",
			// Variable frame rate, otherwise the HLS codec fails.
			// "-vsync", "vfr", // for old versions of ffmpeg
			"-fps_mode:v:0", "vfr",
			// Use H254 encoding.
			"-c:v", "libx264",
			// H264 high profile level 4.2 for maximum support across devices.
			"-profile:v", "high", "-level:v", "4.2",
			// Tee multiple outputs.
			"-f", "tee", "-map", "0:v", "-use_fifo", "1", teeHLS + "|" + teeUDP,
		}...))
		cmd := exec.CommandContext(ctx, program, arg...)
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return nil, errors.Wrap(err, "")
		}
		cmd.Stdout = stdout
		cmd.Stderr = stderr
		cmd.Cancel = func() error {
			_, err := stdin.Write([]byte("q"))
			return err
		}
		cmd.WaitDelay = 10 * time.Second

		return cmd, nil
	}
	return util.NewCmdFn(dir, run)
}

func CreateTables(db *sql.DB) error {
	sqlStrs := []string{
		"CREATE TABLE " + TableRecord + ` (
                        id text PRIMARY KEY,
                        camera text,
                        count text,
                        err text,
                        createAt datetime,
                        stop datetime,
                        cleanup datetime);`,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for _, sqlStr := range sqlStrs {
		if _, err := db.ExecContext(ctx, sqlStr); err != nil {
			return errors.Wrap(err, fmt.Sprintf("\"%s\"", sqlStr))
		}
	}
	return nil
}
