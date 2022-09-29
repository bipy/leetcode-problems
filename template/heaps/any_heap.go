package heaps

import (
	"container/heap"
)

type Heap struct {
	hd *heapData
}

func InitHeap(data []interface{}, less func(i, j interface{}) bool) *Heap {
	h := &Heap{&heapData{data, less}}
	heap.Init(h.hd)
	return h
}

func (h Heap) Top() interface{} {
	return h.hd.data[0]
}

func (h Heap) UpdateTop(x interface{}) {
	h.hd.data[0] = x
	heap.Fix(h.hd, 0)
}

func (h Heap) Push(x interface{}) {
	heap.Push(h.hd, x)
}

func (h Heap) Pop() interface{} {
	return heap.Pop(h.hd)
}

func (h Heap) Size() int {
	return h.hd.Len()
}
