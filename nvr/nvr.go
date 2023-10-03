package nvr

import (
	"bufio"
	_ "embed"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/pkg/errors"
)

//go:embed rtsp.py
var rtspPY string

//go:embed count.py
var countPY string

type Scripts struct {
	RTSP  string
	Count string
}

func NewScripts(dir string) (Scripts, error) {
	s := Scripts{
		RTSP:  filepath.Join(dir, "rtsp.py"),
		Count: filepath.Join(dir, "count.py"),
	}

	if err := os.MkdirAll(dir, 0775); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}

	if err := os.WriteFile(s.RTSP, []byte(rtspPY), 0644); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}
	if err := os.WriteFile(s.Count, []byte(countPY), 0644); err != nil {
		return Scripts{}, errors.Wrap(err, "")
	}

	return s, nil
}

type RTSPProc struct {
	Program string
	Arg     []string
	Cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	stderr io.ReadCloser
}

func NewRTSPProc(scriptPath, name, link, networkInterface, macAddress, username, password string, port int, urlPath string) (*RTSPProc, error) {
	p := &RTSPProc{Program: "python"}
	p.Arg = []string{
		scriptPath,
		"-o=" + name,
		"-l=" + link,
		"-i=" + networkInterface,
		"-m=" + macAddress,
		"-u=" + username,
		"-password=" + password,
		"-port=" + strconv.Itoa(port),
		"-path=" + urlPath,
	}

	p.Cmd = exec.Command(p.Program, p.Arg...)
	var err error
	p.stdin, err = p.Cmd.StdinPipe()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	p.stdout, err = p.Cmd.StdoutPipe()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	p.stderr, err = p.Cmd.StderrPipe()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	go func() {
		scanner := bufio.NewScanner(p.stderr)
		for scanner.Scan() {
			line := scanner.Text()
			log.Printf("%s", line)
		}
		if err := scanner.Err(); err != nil {
			log.Printf("%+v", err)
		}
	}()

	if err := p.Cmd.Start(); err != nil {
		return nil, errors.Wrap(err, "")
	}

	return p, nil
}

type CountProc struct {
}
