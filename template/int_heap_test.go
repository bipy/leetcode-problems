package template

import (
	"container/heap"
	"fmt"
	"testing"
)

func TestIntHeap(t *testing.T) {
	h := &IntHeap{2, 1, 5}
	heap.Init(h)
	heap.Push(h, 3)
	fmt.Printf("minimum: %d\n", (*h)[0])
	for h.Len() > 0 {
		fmt.Printf("%d ", heap.Pop(h))
	}
	fmt.Println()
	// Output:
	// minimum: 1
	// 1 2 3 5
}
