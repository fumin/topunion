package camserver

import (
	"bufio"
	"camserver/util"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"sync"
	"time"

	"github.com/pkg/errors"
)

type CountConfig struct {
	Height int
	Width  int
	Device string
	Mask   struct {
		Enable bool
		Crop   struct {
			X int
			Y int
			W int
		}
		Mask struct {
			Slope int
			Y     int
			H     int
		}
	}
	Yolo struct {
		Weights string
		Size    int
	}
}

type Counter struct {
	sync.RWMutex
	dir string

	cancel context.CancelFunc
	done   chan struct{}

	host string

	gotFirstHost chan struct{}
}

func NewCounter(dir, script string, cfg CountConfig) (*Counter, error) {
	cfgB, err := json.Marshal(cfg)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	cfgStr := string(cfgB)

	ctx, cancel := context.WithCancel(context.Background())
	c := &Counter{dir: dir, cancel: cancel, done: make(chan struct{})}
	c.gotFirstHost = make(chan struct{})
	go func() {
		defer close(c.done)
		for {
			if err := c.run(ctx, script, cfgStr); err != nil {
				log.Printf("%+v", err)
			}
			select {
			case <-ctx.Done():
				return
			default:
			}
		}
	}()
	<-c.gotFirstHost

	return c, nil
}

func (c *Counter) Close() {
	c.cancel()
	<-c.done
}

type CountOutput struct {
	Passed int
}

func (c *Counter) Analyze(ctx context.Context, dst, src string) (CountOutput, error) {
	v := url.Values{}
	v.Set("dst", dst)
	v.Set("src", src)
	urlStr := c.getHost() + "/Analyze?" + v.Encode()
	req, err := http.NewRequestWithContext(ctx, "POST", urlStr, nil)
	if err != nil {
		return CountOutput{}, errors.Wrap(err, "")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return CountOutput{}, errors.Wrap(err, "")
	}
	defer resp.Body.Close()

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return CountOutput{}, errors.Wrap(err, "")
	}
	if resp.StatusCode != http.StatusOK {
		return CountOutput{}, errors.Errorf("body \"%s\"", b)
	}
	var out CountOutput
	if err := json.Unmarshal(b, &out); err != nil {
		return CountOutput{}, errors.Wrap(err, fmt.Sprintf("body \"%s\"", b))
	}
	return out, nil
}

func (c *Counter) run(ctx context.Context, script, cfgStr string) error {
	cmd, stdout, stderr, status, err := util.NewCmd(ctx, c.dir, "python", []string{script, "-c=" + cfgStr})
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer stdout.Close()
	defer stderr.Close()
	if err := cmd.Start(); err != nil {
		return errors.Wrap(err, "")
	}

	var host string
	for i := 0; ; i++ {
		host, err = readHost(stderr.Name())
		if err == nil {
			break
		}
		if i > 60 {
			return errors.Errorf("no host")
		}
		<-time.After(1 * time.Second)
	}
	c.setHost(host)
	defer func() { c.setHost("") }()
	close(c.gotFirstHost)

	if err := cmd.Wait(); err != nil {
		b := []byte(fmt.Sprintf("%+v", err))
		if werr := os.WriteFile(status, b, os.ModePerm); werr != nil {
			return errors.Wrap(werr, "")
		}
	}
	return nil
}

func (c *Counter) getHost() string {
	c.RLock()
	defer c.RUnlock()
	return c.host
}

func (c *Counter) setHost(host string) {
	c.Lock()
	defer c.Unlock()
	c.host = host
}

func readHost(fpath string) (string, error) {
	f, err := os.Open(fpath)
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	defer f.Close()

	lines := make([]string, 0)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return "", errors.Wrap(err, "")
	}

	type logMessage struct {
		Levent string
		Host   string
	}
	for i := len(lines) - 1; i >= 0; i-- {
		line := lines[i]
		var m logMessage
		json.Unmarshal([]byte(line), &m)
		if m.Levent == "host" {
			return m.Host, nil
		}
	}

	return "", errors.Errorf("not found")
}
