package main

import (
	"fmt"
	"time"
)

func main() {
	startTime := time.Now().UnixMilli()
	defer func() { fmt.Println("\nTime Cost:", time.Now().UnixMilli()-startTime) }()
	// TODO
	fmt.Printf("%f", 1e-6)
}