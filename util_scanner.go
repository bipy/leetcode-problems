package leetcode

import (
	"io/ioutil"
	"strings"
)

func GetInput() []string {
	file, err := ioutil.ReadFile("data.in")
	if err != nil {
		panic(err.Error())
	}
	return strings.Split(Bytes2Str(file), "\n")
}
