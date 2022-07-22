package template

import (
	"container/heap"
	"fmt"
	"testing"
)

func TestHeap(t *testing.T) {
	h := &Heap{&Item{priority: 2}, &Item{priority: 1}, &Item{priority: 5}}
	heap.Init(h)
	heap.Push(h, &Item{priority: 3})
	fmt.Printf("minimum: %d\n", (*h)[0].priority)
	for h.Len() > 0 {
		fmt.Printf("%d ", heap.Pop(h).(*Item).priority)
	}
	fmt.Println()
	// Output:
	// minimum: 1
	// 1 2 3 5
}
