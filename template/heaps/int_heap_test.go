package heaps

import (
	"fmt"
	"testing"
)

func TestIntMaxHeap(t *testing.T) {
	arr := []int{2, 1, 5}
	h := InitIntMaxHeap(arr)
	h.Push(3)
	for h.Size() > 0 {
		fmt.Printf("%d ", h.Pop())
	}
}

func TestIntMinHeap(t *testing.T) {
	arr := []int{2, 1, 5}
	h := InitIntMinHeap(arr)
	h.Push(3)
	for h.Size() > 0 {
		fmt.Printf("%d ", h.Pop())
	}
}
