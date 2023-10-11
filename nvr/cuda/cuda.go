package cuda

import (
	"os/exec"
	"strconv"
)

func IsAvailable() bool {
	script := "import torch; print(torch.cuda.is_available())"
	cmd := exec.Command("python", "-c", script)
	stdoutStderr, err := cmd.CombinedOutput()
	if err != nil {
		return false
	}
	isAvailable, err := strconv.ParseBool(string(stdoutStderr))
	if err != nil {
		return false
	}
	return isAvailable
}
