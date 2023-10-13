package cuda

import (
	"os/exec"
	"strconv"
	"strings"
)

func IsAvailable() bool {
	script := "import torch; print(torch.cuda.is_available())"
	cmd := exec.Command("python", "-c", script)
	stdoutStderr, err := cmd.CombinedOutput()
	if err != nil {
		return false
	}
	outerr := strings.TrimSpace(string(stdoutStderr))
	isAvailable, err := strconv.ParseBool(outerr)
	if err != nil {
		return false
	}
	return isAvailable
}
