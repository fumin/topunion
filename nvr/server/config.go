package server

import (
	"nvr"
	"nvr/cuda"
	"time"

	"github.com/pkg/errors"
)

func startVideoWifi(s *Server) (string, error) {
	var record nvr.Record
	rtsp0 := nvr.RTSP{
		Name:             "RedmiNote4X",
		NetworkInterface: "wlx08bfb849ed0a",
		MacAddress:       "4c:49:e3:3a:87:4a",
		Username:         "admin",
		Password:         "0000",
		Port:             8080,
		Path:             "/h264_ulaw.sdp",
	}
	// record.RTSP = append(record.RTSP, rtsp0)
	rtsp1 := nvr.RTSP{
		Name:             "Redmi12C",
		NetworkInterface: "wlx08bfb849ed0a",
		MacAddress:       "f4:1a:9c:67:58:ee",
		Username:         "admin",
		Password:         "0000",
		Port:             8080,
		Path:             "/h264_ulaw.sdp",
	}
	record.RTSP = append(record.RTSP, rtsp1)

	count0 := nvr.Count{Src: rtsp0.Name}
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

	count1 := nvr.Count{Src: rtsp1.Name}
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
	record.Count = append(record.Count, count1)

	id, err := s.startRecord(record)
	if err != nil {
		return "", errors.Wrap(err, "")
	}
	return id, nil
}

func startVideoFile(s *Server, fpath string) (string, error) {
	var record nvr.Record
	rtsp0 := nvr.RTSP{Name: "rtsp0", Link: fpath}
	record.RTSP = append(record.RTSP, rtsp0)

	count0 := nvr.Count{Src: rtsp0.Name}
	count0.Config.AI.Smart = false
	count0.Config.AI.Device = "cpu"
	if cuda.IsAvailable() {
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
	go func() {
		<-time.After((1*60 + 3) * time.Second)
		s.stopRecord(id)
	}()

	return id, nil
}
