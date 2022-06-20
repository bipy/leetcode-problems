package leetcode

import (
	"fmt"
	"testing"
)

func TestStrTo2DStrSlice(t *testing.T) {
	fmt.Println(StrTo2DStrSlice("[[\"ac\",\"ab\",\"zc\",\"zb\"],[\"ac\",\"ab\",\"zc\",\"zb\"]]"))
}

func TestStrTo2DIntSlice(t *testing.T) {
	fmt.Println(StrTo2DIntSlice("[[12,10,15],[20,23,8],[21,7,1],[8,1,13],[9,10,25],[5,3,2]]"))

}

func TestStrToStrSlice(t *testing.T) {
	fmt.Println(StrToStrSlice("[\"ac\",\"ab\",\"zc\",\"zb\"]"))
}

func TestStrToIntSlice(t *testing.T) {
	fmt.Println(StrToIntSlice("[1,2,3,4]"))
}
