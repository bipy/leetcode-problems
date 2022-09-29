package leetcode

import (
	"os"
	"strings"
)

func GetInput() []string {
	file, err := os.ReadFile("data.in")
	if err != nil {
		panic(err.Error())
	}
	return strings.Split(Bytes2Str(file), "\n")
}
