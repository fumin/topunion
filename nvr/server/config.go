package server

import (
	"fmt"
	"nvr"
	"nvr/cuda"

	"github.com/pkg/errors"
)

func startVideoWifi(s *Server) (string, error) {
	var record nvr.Record
	camera0 := nvr.Camera{
		Name:             "RedmiNote4X",
		Input:            []string{"rtsp://admin:0000@{{.IP}}:8080/h264_ulaw.sdp"},
		NetworkInterface: "wlx08bfb849ed0a",
		MacAddress:       "4c:49:e3:3a:87:4a",
	}
	// record.Camera = append(record.Camera, camera0)
	camera1 := nvr.Camera{
		Name:             "Redmi12C",
		Input:            []string{"rtsp://admin:0000@{{.IP}}:8080/h264_ulaw.sdp"},
		NetworkInterface: "wlx08bfb849ed0a",
		MacAddress:       "f4:1a:9c:67:58:ee",
	}
	// record.Camera = append(record.Camera, camera1)
	camera2 := nvr.Camera{
		Name:             "FuminPhone",
		Input:            []string{"-rtsp_transport", "tcp", "-i", "rtsp://admin:0000@{{.IP}}:8080/h264_ulaw.sdp"},
		NetworkInterface: "wlx08bfb849ed0a",
		MacAddress:       "94:7b:ae:94:ca:80",
	}
	record.Camera = append(record.Camera, camera2)

	count0 := nvr.Count{Src: camera0.Name}
	count0.Config.AI.Smart = true
	count0.Config.AI.Device = "cuda:0"
	count0.Config.AI.Mask.Enable = false
	count0.Config.AI.Mask.Crop.X = 100
	count0.Config.AI.Mask.Crop.Y = 0
	count0.Config.AI.Mask.Crop.W = 1700
	count0.Config.AI.Mask.Mask.Slope = 10
	count0.Config.AI.Mask.Mask.Y = 500
	count0.Config.AI.Mask.Mask.H = 200
	count0.Config.AI.Yolo.Weights = "yolo_best.pt"
	count0.Config.AI.Yolo.Size = 640
	// record.Count = append(record.Count, count0)

	count1 := nvr.Count{Src: camera1.Name}
	count1.Config.AI.Smart = true
	count1.Config.AI.Device = "cuda:0"
	count1.Config.AI.Mask.Enable = false
	count1.Config.AI.Mask.Crop.X = 100
	count1.Config.AI.Mask.Crop.Y = 0
	count1.Config.AI.Mask.Crop.W = 1700
	count1.Config.AI.Mask.Mask.Slope = 10
	count1.Config.AI.Mask.Mask.Y = 500
	count1.Config.AI.Mask.Mask.H = 200
	count1.Config.AI.Yolo.Weights = "yolo_best.pt"
	count1.Config.AI.Yolo.Size = 640
	// record.Count = append(record.Count, count1)

	id, err := s.startRecord(record)
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	return id, nil
}

func startVideoFile(s *Server, fpath string) (string, error) {
	var record nvr.Record
	camera0 := nvr.Camera{Name: "camera0", Input: []string{fpath}, Repeat: 1}
	record.Camera = append(record.Camera, camera0)

	count0 := nvr.Count{Src: camera0.Name}
	count0.Config.AI.Device = "cpu"
	if cuda.IsAvailable() {
		count0.Config.AI.Smart = true
		count0.Config.AI.Device = "cuda:0"
	}
	count0.Config.AI.Mask.Enable = true
	count0.Config.AI.Mask.Crop.X = 100
	count0.Config.AI.Mask.Crop.Y = 0
	count0.Config.AI.Mask.Crop.W = 1700
	count0.Config.AI.Mask.Mask.Slope = 10
	count0.Config.AI.Mask.Mask.Y = 500
	count0.Config.AI.Mask.Mask.H = 200
	count0.Config.AI.Yolo.Weights = "yolo_best.pt"
	count0.Config.AI.Yolo.Size = 640
	record.Count = append(record.Count, count0)

	id, err := s.startRecord(record)
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	return id, nil
}

func startSMPTE(s *Server) (string, error) {
	var record nvr.Record
	for i := 0; i < 2; i++ {
		camera := nvr.Camera{Name: fmt.Sprintf("camera%d", i), Input: []string{"-f", "lavfi", "-i", "smptebars"}}
		record.Camera = append(record.Camera, camera)
		count := nvr.Count{Src: camera.Name}
		count.Config.AI.Smart = false
		count.Config.AI.Device = "cpu"
		count.Config.AI.Mask.Enable = false
		record.Count = append(record.Count, count)
	}

	id, err := s.startRecord(record)
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	return id, nil
}
