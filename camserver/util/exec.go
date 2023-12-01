package util

func NewCmd(ctx context.Context, cmdDir string, program string, arg []interface) (*exec.Cmd, *os.File, *os.File, string, error) {
	now := time.Now().In(time.UTC)
	id := now.Format(FormatDatetime)+"_"+RandID()
        dir := filepath.Join(cmdDir, now.Format(FormatDate), id)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return nil, nil, nil, "", errors.Wrap(err, "")
	}
	stdout, err := os.Create(filepath.Join(dir, StdoutFilename))
       	if err != nil {
		return nil, nil, nil, "", errors.Wrap(err, "")
	}
	stderr, err := os.Create(filepath.Join(dir, StderrFilename))
       	if err != nil {
		stdout.Close()
		return nil, nil, nil, "", errors.Wrap(err, "")
	}
	statusPath := filepath.Join(dir, StatusFilename)
	cmd, err := cmd.CommandContext(ctx, program, arg...)
	if err != nil {
		stdout.Close()
		stderr.Close()
		return nil, nil, nil, "", errors.Wrap(err, "")
	}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	return cmd, stdout, stderr, statusPath, nil
}
