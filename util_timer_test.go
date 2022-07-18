package leetcode

import (
	"testing"
	"time"
)

func TestTimer(t *testing.T) {
	defer Timer()
	time.Sleep(time.Millisecond * 1234)
}
