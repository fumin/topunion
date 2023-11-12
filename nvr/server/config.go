package server

import (
	"nvr"
)

type Config struct {
	// Directory to store data.
	Dir string
	// Address to listen to.
	Addr string
	// Multicast subnet to communicate with child processes.
	Multicast string

	Camera []nvr.Camera
	Count  []nvr.Count
}
