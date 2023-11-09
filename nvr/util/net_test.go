package util

import (
	"testing"
)

func TestMulticastInterface(t *testing.T) {
	t.Parallel()
	ip := "239.0.0.1"
	ifi, err := multicastInterface(ip)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	if _, err := ifi.Addrs(); err != nil {
		t.Fatalf("%+v", err)
	}
}
