package heaps

import (
	"container/heap"
)

type heapData struct {
	data []interface{}
	less func(i, j interface{}) bool
}

type Heap struct {
	hd *heapData
}

func InitHeap(data []interface{}, less func(i, j interface{}) bool) *Heap {
	h := &Heap{&heapData{data, less}}
	heap.Init(h.hd)
	return h
}

func (hd heapData) Len() int {
	return len(hd.data)
}

func (hd heapData) Swap(i, j int) {
	hd.data[i], hd.data[j] = hd.data[j], hd.data[i]
}

func (hd heapData) Less(i, j int) bool {
	return hd.less(hd.data[i], hd.data[j])
}

func (hd *heapData) Push(x interface{}) {
	hd.data = append(hd.data, x)
}

func (hd *heapData) Pop() interface{} {
	n := len(hd.data)
	x := hd.data[n-1]
	hd.data = hd.data[:n-1]
	return x
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
