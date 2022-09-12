package segment_tree

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSegmentTree(t *testing.T) {
	arr := make([]int, 10)
	arr[0] = 10
	st := InitSegmentTree(arr, func(a, b int) int {
		return a + b
	})
	assert.Equal(t, 10, st.Query(0, 10))
	assert.Equal(t, 10, st.Query(0, 1))
	assert.Equal(t, 10, st.Query(0, 0))
	assert.Equal(t, 0, st.Query(1, 1))
	st.Update(2, 5)
	st.Update(3, 6)
	st.Update(7, 10)
	assert.Equal(t, 11, st.Query(2, 5))
	assert.Equal(t, 21, st.Query(2, 7))
	assert.Equal(t, 31, st.QueryAll())
}

func TestSegmentTreeMax(t *testing.T) {
	arr := make([]int, 100001)
	st := InitSegmentTree(arr, func(a, b int) int {
		if a > b {
			return a
		}
		return b
	})
	st.Update(1, 1)
	st.Update(100, 2)
	st.Update(500, 3)
	st.Update(100000, 4)
	assert.Equal(t, 1, st.Query(0, 1))
	assert.Equal(t, 2, st.Query(0, 100))
	assert.Equal(t, 3, st.Query(0, 500))
	assert.Equal(t, 4, st.Query(0, 100000))
	assert.Equal(t, 3, st.Query(0, 99999))
}
