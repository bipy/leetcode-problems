package algo

import (
	"github.com/stretchr/testify/assert"
	"math/rand"
	"sort"
	"testing"
	"time"
)

func TestQuickSort(t *testing.T) {
	arrs := generateRandomSlices()
	for _, arr := range arrs {
		QuickSort(arr)
		assert.True(t, sort.IntsAreSorted(arr))
	}
}

func TestRandomizedQuickSort(t *testing.T) {
	arrs := generateRandomSlices()
	for _, arr := range arrs {
		RandomizedQuickSort(arr)
		assert.True(t, sort.IntsAreSorted(arr))
	}
}

func TestMergeSort(t *testing.T) {
	arrs := generateRandomSlices()
	for _, arr := range arrs {
		MergeSort(arr)
		assert.True(t, sort.IntsAreSorted(arr))
	}
}

func TestBubbleSortSort(t *testing.T) {
	arrs := generateRandomSlices()
	for _, arr := range arrs {
		BubbleSort(sort.IntSlice(arr))
		assert.True(t, sort.IntsAreSorted(arr))
	}
}

func TestHeapSort(t *testing.T) {
	arrs := generateRandomSlices()
	for _, arr := range arrs {
		HeapSort(arr)
		assert.True(t, sort.IntsAreSorted(arr))
	}
}

func TestInsertionSort(t *testing.T) {
	arrs := generateRandomSlices()
	for _, arr := range arrs {
		InsertionSort(arr)
		assert.True(t, sort.IntsAreSorted(arr))
	}
}

func TestSelectionSort(t *testing.T) {
	arrs := generateRandomSlices()
	for _, arr := range arrs {
		SelectionSort(arr)
		assert.True(t, sort.IntsAreSorted(arr))
	}
}

func TestShellSort(t *testing.T) {
	arrs := generateRandomSlices()
	for _, arr := range arrs {
		ShellSort(arr)
		assert.True(t, sort.IntsAreSorted(arr))
	}
}

func generateRandomSlices() (rt [][]int) {
	rand.Seed(time.Now().UnixNano())
	var lengths = []int{1, 2, 3, 4, 5, 10, 100, 1000, 1e4, 1e4, 1e4}
	for _, length := range lengths {
		temp := make([]int, length)
		for i := range temp {
			temp[i] = rand.Intn(length * 10)
		}
		rt = append(rt, temp)
	}
	sort.Ints(rt[9])
	sort.Sort(sort.Reverse(sort.IntSlice(rt[10])))
	return
}
