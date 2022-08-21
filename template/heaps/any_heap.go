package heaps

import (
	"container/heap"
)

type Heap struct {
	Data []interface{}
	less func(i, j interface{}) bool
}

func InitHeap(data []interface{}, less func(i, j interface{}) bool) *Heap {
	h := &Heap{data, less}
	heap.Init(h)
	return h
}

func (h Heap) Len() int {
	return len(h.Data)
}

func (h Heap) Swap(i, j int) {
	h.Data[i], h.Data[j] = h.Data[j], h.Data[i]
}

func (h Heap) Less(i, j int) bool {
	return h.less(h.Data[i], h.Data[j])
}

func (h *Heap) Push(x interface{}) {
	h.Data = append(h.Data, x)
}

func (h *Heap) Pop() interface{} {
	n := len(h.Data)
	x := h.Data[n-1]
	h.Data = h.Data[:n-1]
	return x
}

func (h Heap) Top() interface{} {
	return h.Data[0]
}

func (h *Heap) Update(idx int, x interface{}) {
	h.Data[idx] = x
	heap.Fix(h, idx)
}
