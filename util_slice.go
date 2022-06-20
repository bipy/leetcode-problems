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
		return nil
	}
	rt := strings.Split(input, ",")
	for i := range rt {
		rt[i] = strings.TrimPrefix(rt[i], "\"")
		rt[i] = strings.TrimSuffix(rt[i], "\"")
	}
	return rt
}

func StrToIntSlice(input string) (rt []int) {
	input = strings.TrimPrefix(input, "[")
	input = strings.TrimSuffix(input, "]")
	if input == "" {
		return nil
	}
	s := strings.Split(input, ",")
	for i := range s {
		d, err := strconv.Atoi(s[i])
		if err != nil {
			fmt.Println("Error")
		}
		rt = append(rt, d)
	}
	return
}

func StrTo2DIntSlice(input string) (rt [][]int) {
	input = strings.TrimPrefix(input, "[[")
	input = strings.TrimSuffix(input, "]]")
	s := strings.Split(input, "],[")
	for i := range s {
		if is := StrToIntSlice(s[i]); is != nil {
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
		if ss := StrToStrSlice(s[i]); ss != nil {
			rt = append(rt, ss)
		}
	}
	return
}
