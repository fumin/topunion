package util

import (
	"context"
	"fmt"
	"net"
	"os/exec"
	"strings"
	"time"

	"github.com/pkg/errors"
)

func Inc(ip net.IP) {
	for j := len(ip) - 1; j >= 0; j-- {
		ip[j]++
		if ip[j] > 0 {
			break
		}
	}
}

func Loopback() (*net.Interface, error) {
	ifis, err := net.Interfaces()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	for _, ifi := range ifis {
		addrs, err := ifi.Addrs()
		if err != nil {
			continue
		}
		for _, addr := range addrs {
			if strings.HasPrefix(addr.String(), "127.0.0.1") {
				return &ifi, nil
			}
		}
	}
	return nil, ErrNotFound
}

func MulticastInterface(addr string) (*net.Interface, error) {
	// Start multicast sender.
	cmdNameArgs := strings.Split(fmt.Sprintf("ffmpeg -f lavfi -i smptebars -c:v libx264 -f tee -map 0:v [f=mpegts]udp://%s:10000?pkt_size=%d", addr, VLCUDPLen), " ")
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, cmdNameArgs[0], cmdNameArgs[1:]...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	if err := cmd.Start(); err != nil {
		return nil, errors.Wrap(err, "")
	}

	// Get the interface of the multicast sender.
	ifi, err := multicastInterface(addr)
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	// Stop multicast sender
	if _, err := stdin.Write([]byte{'q'}); err != nil {
		return nil, errors.Wrap(err, "")
	}
	if err := cmd.Wait(); err != nil {
		return nil, errors.Wrap(err, "")
	}

	return ifi, nil
}

func multicastInterface(addr string) (*net.Interface, error) {
	ifis, err := net.Interfaces()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}
	for _, ifi := range ifis {
		addrs, err := ifi.MulticastAddrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			// log.Printf("%s %s", a, ifi.Name)
			if a.String() == addr {
				return &ifi, nil
			}
		}
	}
	return nil, ErrNotFound
}
