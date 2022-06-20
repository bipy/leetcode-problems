package leetcode

import (
	"fmt"
	"time"
)

var startTime = time.Now().UnixMilli()

func Timer() {
	fmt.Println("\n\033[31mTime Cost:", time.Now().UnixMilli()-startTime, "ms\033[0m")
}
