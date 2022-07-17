package template

import (
	"container/heap"
	"fmt"
	"testing"
)

func TestHeap(t *testing.T) {
	h := &Heap{&HeapItem{y: 2}, &HeapItem{y: 1}, &HeapItem{y: 5}}
	heap.Init(h)
	heap.Push(h, &HeapItem{y: 3})
	fmt.Printf("minimum: %d\n", (*h)[0].y)
	for h.Len() > 0 {
		fmt.Printf("%d ", heap.Pop(h).(*HeapItem).y)
	}
	fmt.Println()
	// Output:
	// minimum: 1
	// 1 2 3 5
}
