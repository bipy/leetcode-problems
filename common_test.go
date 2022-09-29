package leetcode

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"math/rand"
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

func TestZip(t *testing.T) {
	a := []int{1, 2, 3, 4}
	fmt.Println(Zip(a, a, a, a, a))
}

func TestReduce(t *testing.T) {
	a := []int{6, 12, 60, 120}
	fmt.Println(Reduce(a, GCD))
}

func TestSplit(t *testing.T) {
	arr := make([]int, 10000)
	for i := range arr {
		arr[i] = i
	}
	rand.Shuffle(len(arr), func(i, j int) {
		arr[i], arr[j] = arr[j], arr[i]
	})
	less, greater := SplitT[int](arr, func(x int) bool {
		return x <= 5000
	})
	assert.Equal(t, 5000, maxOf(less...))
	assert.Equal(t, 5001, minOf(greater...))
}

func TestSplit1(t *testing.T) {
	arr := make([]int, 10000)
	for i := range arr {
		arr[i] = i
	}
	rand.Shuffle(len(arr), func(i, j int) {
		arr[i], arr[j] = arr[j], arr[i]
	})
	even, odd := SplitT[int](arr, func(x int) bool {
		return x%2 == 0
	})
	for _, v := range even {
		if v%2 == 1 {
			assert.FailNow(t, "even")
		}
	}
	for _, v := range odd {
		if v%2 == 0 {
			assert.FailNow(t, "odd")
		}
	}
}
