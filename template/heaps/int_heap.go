package heaps

import "container/heap"

type IntHeap struct {
	hd *heapData
}

func initIntHeap(data []int, less func(i, j interface{}) bool) *IntHeap {
	t := make([]interface{}, len(data))
	for i := range t {
		t[i] = data[i]
	}
	h := &IntHeap{&heapData{t, less}}
	heap.Init(h.hd)
	return h
}

func InitIntMaxHeap(data []int) *IntHeap {
	return initIntHeap(data, func(i, j interface{}) bool {
		return i.(int) > j.(int)
	})
}

func InitIntMinHeap(data []int) *IntHeap {
	return initIntHeap(data, func(i, j interface{}) bool {
		return i.(int) < j.(int)
	})
}

func (h IntHeap) Top() int {
	return h.hd.data[0].(int)
}

func (h IntHeap) UpdateTop(x int) {
	h.hd.data[0] = x
	heap.Fix(h.hd, 0)
}

func (h IntHeap) Push(x int) {
	heap.Push(h.hd, x)
}

func (h IntHeap) Pop() int {
	return heap.Pop(h.hd).(int)
}

func (h IntHeap) Size() int {
	return h.hd.Len()
}
