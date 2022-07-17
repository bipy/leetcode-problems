package template

import (
	"fmt"
	"testing"
)

func TestSegmentTree(t *testing.T) {
	st := InitSegmentTree(make([]int, 10))
	fmt.Println(st.QuerySum(1, 10))
	st.Add(2, 5)
	st.Add(3, 6)
	st.Add(7, 10)
	fmt.Println(st.QuerySum(1, 10))
}
