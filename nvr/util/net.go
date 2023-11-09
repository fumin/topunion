package util

import (
	"fmt"
	"net"

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

func BroadcastAddr(ipnet *net.IPNet) net.IP {
	broadcast := make([]byte, len(ipnet.IP))
	copy(broadcast, ipnet.IP)

	ones, bits := ipnet.Mask.Size()
	netBits := bits - ones
	for i := 0; i < netBits; i++ {
		byteIdx := i / 8
		bitIdx := i % 8
		broadcast[len(broadcast)-1-byteIdx] |= (1 << bitIdx)
	}

	return broadcast
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
			if net.ParseIP(addr.String()).IsLoopback() {
				return &ifi, nil
			}
		}
	}
	return nil, ErrNotFound
}

func MulticastAddrs() (map[string][]net.Interface, error) {
	ifis, err := net.Interfaces()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	mAddrs := make(map[string][]net.Interface)
	for _, ifi := range ifis {
		addrs, err := ifi.MulticastAddrs()
		if err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("%#v", ifi))
		}
		for _, a := range addrs {
			s := a.String()
			mAddrs[s] = append(mAddrs[s], ifi)
		}
	}
	return mAddrs, nil
}
