package util

import (
	"net"
	"testing"
)

func TestBroadcastAddr(t *testing.T) {
	type testcase struct {
		net       string
		broadcast string
	}
	cases := []testcase{
		{net: "62.76.47.18/28", broadcast: "62.76.47.31"},
		{net: "62.76.47.231/28", broadcast: "62.76.47.239"},
		{net: "62.76.45.231/22", broadcast: "62.76.47.255"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.net, func(t *testing.T) {
			t.Parallel()
			_, ipnet, err := net.ParseCIDR(tc.net)
			if err != nil {
				t.Fatalf("%+v", err)
			}
			broadcast := BroadcastAddr(ipnet)
			if !broadcast.Equal(net.ParseIP(tc.broadcast)) {
				t.Fatalf("%s %#v", broadcast, tc)
			}
		})
	}
}

// func TestMulticastAddrs(t *testing.T) {
// 	t.Parallel()
//
// 	ip := "239.0.0.1"
// 	// Start multicast sender.
// 	cmdNameArgs := strings.Split(fmt.Sprintf("ffmpeg -f lavfi -i smptebars -c:v libx264 -f tee -map 0:v [f=mpegts]udp://%s:10000?pkt_size=%d", ip, VLCUDPLen), " ")
// 	const waitSecs = 3
// 	ctx, cancel := context.WithTimeout(context.Background(), waitSecs*2*time.Second)
// 	defer cancel()
// 	cmd := exec.CommandContext(ctx, cmdNameArgs[0], cmdNameArgs[1:]...)
// 	stdin, err := cmd.StdinPipe()
// 	if err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// 	if err := cmd.Start(); err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// 	// Wait a while for the multicast packets to be sent out.
// 	<-time.After(waitSecs * time.Second)
//
// 	addrs, err := MulticastAddrs()
// 	if err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// 	for addr, ifis := range addrs {
// 		names := make([]string, 0, len(ifis))
// 		for _, ifi := range ifis {
// 			names = append(names, ifi.Name)
// 		}
// 		t.Logf("%s %#v", addr, names)
// 	}
//
// 	// Stop multicast sender
// 	if _, err := stdin.Write([]byte{'q'}); err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// 	if err := cmd.Wait(); err != nil {
// 		t.Fatalf("%+v", err)
// 	}
// }
