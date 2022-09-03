package leetcode

import (
	"fmt"
	"strconv"
	"strings"
)

func StrToStrSlice(input string) []string {
	input = strings.TrimPrefix(input, "[")
	input = strings.TrimSuffix(input, "]")
	if input == "" {
		return []string{}
	}
	rt := strings.Split(input, ",")
	for i := range rt {
		rt[i] = strings.TrimPrefix(rt[i], "\"")
		rt[i] = strings.TrimSuffix(rt[i], "\"")
	}
	return rt
}

func StrToIntSlice(input string) []int {
	input = strings.TrimPrefix(input, "[")
	input = strings.TrimSuffix(input, "]")
	if input == "" {
		return []int{}
	}
	s := strings.Split(input, ",")
	rt := make([]int, len(s))
	for i := range s {
		d, err := strconv.Atoi(s[i])
		if err != nil {
			fmt.Println("Error", err.Error())
		}
		rt[i] = d
	}
	return rt
}

func StrToByteSlice(input string) []byte {
	input = strings.TrimPrefix(input, "[")
	input = strings.TrimSuffix(input, "]")
	if input == "" {
		return []byte{}
	}
	rt := strings.Split(input, ",")
	b := make([]byte, len(rt))
	for i := range rt {
		rt[i] = strings.TrimPrefix(rt[i], "\"")
		rt[i] = strings.TrimSuffix(rt[i], "\"")
		b[i] = rt[i][0]
	}
	return b
}

func StrTo2DIntSlice(input string) (rt [][]int) {
	input = strings.TrimPrefix(input, "[[")
	input = strings.TrimSuffix(input, "]]")
	s := strings.Split(input, "],[")
	for i := range s {
		if is := StrToIntSlice(s[i]); len(is) != 0 {
			rt = append(rt, is)
		}
	}
	return
}

func StrTo2DStrSlice(input string) (rt [][]string) {
	input = strings.TrimPrefix(input, "[[")
	input = strings.TrimSuffix(input, "]]")
	s := strings.Split(input, "],[")
	for i := range s {
		if ss := StrToStrSlice(s[i]); len(ss) != 0 {
			rt = append(rt, ss)
		}
	}
	return
}

func StrTo2DByteSlice(input string) (rt [][]byte) {
	input = strings.TrimPrefix(input, "[[")
	input = strings.TrimSuffix(input, "]]")
	s := strings.Split(input, "],[")
	for i := range s {
		if bs := StrToByteSlice(s[i]); len(bs) != 0 {
			rt = append(rt, bs)
		}
	}
	return
}
