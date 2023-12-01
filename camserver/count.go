package camserver

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
	sync.Mutex
	cancel context.CancelFunc
	host string
}

func newCounter(script string, cfg CountConfig) (*Counter, error) {
	cfgB, err := json.Marshal(cfg)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	cfgStr := string(cfgB)

	c := &Counter{}
	go func(){
		for {
			ctx, cancel := context.WithCancel(context.Background())
			setHost := func(host string) {
				c.Lock()
				defer c.Unlock()
				c.cancel = cancel
				c.host = host
			}
			runCounter(ctx, cfgStr, setHost)
		}
	}()

	return c, nil
}

func runCounter(ctx context.Context, cfgStr string, setHost func(string)) error {
	cmd, stdout, stderr, status, err := util.NewCmd(ctx, CounterDir, "python", []string{"-c="+cfgStr})
	if err != nil {
		return errors.Wrap(err, "")
	}
	defer stdout.Close()
	defer stderr.Close()
	if err := cmd.Start(); err != nil {
		return errors.Wrap(err, "")
	}

	var host string
	for {
		host, err = readHost(stderr.Name())
		if err == nil {
			break
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(1):
		}
	}
	setHost(host)

	if err := cmd.Wait(); err != nil {
		os.WriteFile(status, []byte(fmt.Sprintf("%+v", err)))
		return errors.Wrap(err, "")
	}
	return nil
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
		Host string
	}
	for i := len(lines)-1; i >= 0; i-- {
		line := lines[i]
		var m logMessage
		json.Unmarshal([]byte(line), &m)
		if Levent == "host" {
			return m.Host, nil
		}
	}

	return "", errors.Errorf("not found")
}
