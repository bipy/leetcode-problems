package leetcode

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_maxOf(t *testing.T) {
	assert.Equal(t, 10, maxOf(1, 2, 4, 8, 10, 4, 1))
	assert.Equal(t, 0x7fffffff, maxOf(1, 2, 4, 8, 10, 4, 1, 0x7fffffff))
}

func Test_minOf(t *testing.T) {
	assert.Equal(t, 1, minOf(1, 2, 4, 8, 10, 4, 1))
	assert.Equal(t, -1<<31, minOf(1, 2, 4, 8, 10, 4, 1, -1<<31))
}

func Test_maxOfSlice(t *testing.T) {
	arr := []int{1, 2, 4, 8, 10, 4, 1}
	idx := maxOfSlice(arr, func(i, j int) bool {
		return arr[i] < arr[j]
	})
	assert.Equal(t, 10, arr[idx])
}

func Test_minOfSlice(t *testing.T) {
	arr := []int{1, 2, 4, 8, 10, 4, 1}
	idx := minOfSlice(arr, func(i, j int) bool {
		return arr[i] < arr[j]
	})
	assert.Equal(t, 1, arr[idx])
}
