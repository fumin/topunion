package arp

import (
	"bufio"
	"bytes"
	"fmt"
	"os/exec"
	"strings"

	"github.com/pkg/errors"
)

type Hardware struct {
	IP         string
	MacAddress string
}

func Scan(networkInterface string) (map[string]Hardware, error) {
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

	hws := make(map[string]Hardware)
	scanner := bufio.NewScanner(bytes.NewBuffer(b))
	for scanner.Scan() {
		line := scanner.Text()
		cols := strings.Split(line, "\n")
		if len(cols) < 2 {
			return nil, errors.Wrap(err, fmt.Sprintf("%#v %s", cols, b))
		}

		var hw Hardware
		hw.IP = cols[0]
		hw.MacAddress = cols[1]
		hws[hw.MacAddress] = hw
	}
	if scanner.Err(); err != nil {
		return nil, errors.Wrap(err, "")
	}
	return hws, nil
}
