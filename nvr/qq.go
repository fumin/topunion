package main

import (
	"log"
	"time"
)

func main() {
	loc, err := time.LoadLocation("Asia/Taipei")
	if err != nil {
		log.Fatalf("%+v", err)
	}
	log.Printf("%#v", loc)
}
