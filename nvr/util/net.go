package util

import (
	"fmt"
	"log"
	"net"

	"github.com/pkg/errors"
)

var (
	LoopbackInterface = loopbackMust()
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

func Loopback() ([]net.Interface, error) {
	ifis, err := net.Interfaces()
	if err != nil {
		return nil, errors.Wrap(err, "")
	}

	los := make([]net.Interface, 0)
	for _, ifi := range ifis {
		addrs, err := ifi.Addrs()
		if err != nil {
			return nil, errors.Wrap(err, "")
		}

		for _, addr := range addrs {
			if parseIP(addr.String()).IsLoopback() {
				los = append(los, ifi)
				break
			}
		}
	}
	return los, nil
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

func parseIP(addr string) net.IP {
	if ip := net.ParseIP(addr); len(ip) > 0 {
		return ip
	}
	if ip, _, err := net.ParseCIDR(addr); err == nil {
		return ip
	}
	return nil
}

func loopbackMust() *net.Interface {
	los, err := Loopback()
	if err != nil {
		log.Fatalf("%+v", err)
	}
	if len(los) == 0 {
		log.Fatalf("no loopback")
	}
	return &los[0]
}
