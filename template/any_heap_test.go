package template

import (
	"container/heap"
	"fmt"
	"testing"
)

func TestHeap(t *testing.T) {
	h := &Heap{node{y: 2}, node{y: 1}, node{y: 5}}
	heap.Init(h)
	heap.Push(h, node{y: 3})
	fmt.Printf("minimum: %d\n", (*h)[0].y)
	for h.Len() > 0 {
		fmt.Printf("%d ", heap.Pop(h).(node).y)
	}
	fmt.Println()
	// Output:
	// minimum: 1
	// 1 2 3 5
}
