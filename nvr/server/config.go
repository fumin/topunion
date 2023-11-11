package server

import (
	"fmt"
	"nvr"
	"nvr/cuda"
)

func recordCamera() nvr.Record {
	var record nvr.Record

	camera0 := nvr.Camera{
		Name:             "RedmiNote4X",
		Input:            []string{"rtsp://admin:0000@{{.IP}}:8080/h264_ulaw.sdp"},
		NetworkInterface: "wlx08bfb849ed0a",
		MacAddress:       "4c:49:e3:3a:87:4a",
	}
	record.Camera = append(record.Camera, camera0)
	count0 := nvr.Count{Src: camera0.Name}
	count0.Config.AI.Smart = true
	count0.Config.AI.Device = "cuda:0"
	count0.Config.AI.Mask.Enable = false
	count0.Config.AI.Yolo.Weights = "yolo_best.pt"
	count0.Config.AI.Yolo.Size = 640
	record.Count = append(record.Count, count0)

	// camera1 := nvr.Camera{
	// 	Name:             "Redmi12C",
	// 	Input:            []string{"rtsp://admin:0000@{{.IP}}:8080/h264_ulaw.sdp"},
	// 	NetworkInterface: "wlx08bfb849ed0a",
	// 	MacAddress:       "f4:1a:9c:67:58:ee",
	// }
	// record.Camera = append(record.Camera, camera1)
	// count1 := nvr.Count{Src: camera1.Name}
	// count1.Config.AI.Smart = true
	// count1.Config.AI.Device = "cuda:0"
	// count1.Config.AI.Mask.Enable = false
	// count1.Config.AI.Yolo.Weights = "yolo_best.pt"
	// count1.Config.AI.Yolo.Size = 640
	// record.Count = append(record.Count, count1)

	// camera2 := nvr.Camera{
	// 	Name:             "FuminPhone",
	// 	Input:            []string{"-rtsp_transport", "tcp", "-i", "rtsp://admin:0000@{{.IP}}:8080/h264_ulaw.sdp"},
	// 	NetworkInterface: "wlx08bfb849ed0a",
	// 	MacAddress:       "94:7b:ae:94:ca:80",
	// }
	// record.Camera = append(record.Camera, camera2)
	// count2 := nvr.Count{Src: camera2.Name}
	// count2.Config.AI.Smart = true
	// count2.Config.AI.Device = "cuda:0"
	// count2.Config.AI.Mask.Enable = false
	// count2.Config.AI.Yolo.Weights = "yolo_best.pt"
	// count2.Config.AI.Yolo.Size = 640
	// record.Count = append(record.Count, count2)

	// video := nvr.Camera{Name: "video", Input: []string{"-stream_loop", "-1", "-re", "-i", "sample/shilin20230826.mp4"}}
	// record.Camera = append(record.Camera, video)
	// countVideo := nvr.Count{Src: video.Name}
	// countVideo.Config.AI.Smart = true
	// countVideo.Config.AI.Device = "cuda:0"
	// countVideo.Config.AI.Mask.Enable = true
	// countVideo.Config.AI.Mask.Crop.X = 100
	// countVideo.Config.AI.Mask.Crop.Y = 0
	// countVideo.Config.AI.Mask.Crop.W = 1700
	// countVideo.Config.AI.Mask.Mask.Slope = 10
	// countVideo.Config.AI.Mask.Mask.Y = 500
	// countVideo.Config.AI.Mask.Mask.H = 200
	// countVideo.Config.AI.Yolo.Weights = "yolo_best.pt"
	// countVideo.Config.AI.Yolo.Size = 640
	// record.Count = append(record.Count, countVideo)

	return record
}

func recordVideoFile(fpath string) nvr.Record {
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

	return record
}

func recordSMPTE() nvr.Record {
	var record nvr.Record
	for i := 0; i < 2; i++ {
		camera := nvr.Camera{Name: fmt.Sprintf("camera%d", i), Input: []string{"-re", "-f", "lavfi", "-i", "smptebars"}}
		record.Camera = append(record.Camera, camera)
		count := nvr.Count{Src: camera.Name}
		count.Config.AI.Smart = false
		count.Config.AI.Device = "cpu"
		count.Config.AI.Mask.Enable = false
		record.Count = append(record.Count, count)
	}

	return record
}
