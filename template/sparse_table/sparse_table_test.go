package sparse_table

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSparseTable(t *testing.T) {
	arr := make([]int, 10)
	arr[0] = 10
	arr[5] = 20
	st := InitSparseTable(arr, func(a, b int) int {
		return max(a, b)
	})
	assert.Equal(t, 20, st.Query(0, 10))
	assert.Equal(t, 10, st.Query(0, 1))
	assert.Equal(t, 10, st.Query(0, 1))
	assert.Equal(t, 0, st.Query(1, 2))
	assert.Equal(t, 0, st.Query(2, 5))
	assert.Equal(t, 20, st.Query(2, 6))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
