package template

import "container/heap"

type Item struct {
	value    any
	priority int
	index    int
}

type PriorityQueue []*Item

func (pq PriorityQueue) Len() int {
	return len(pq)
}

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].priority > pq[j].priority
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x any) {
	x.(*Item).index = len(*pq)
	*pq = append(*pq, x.(*Item))
}

// Pop panic if PriorityQueue is empty
func (pq *PriorityQueue) Pop() (v any) {
	*pq, v = (*pq)[:pq.Len()-1], (*pq)[pq.Len()-1]
	return
}

func (pq *PriorityQueue) Update(item *Item, value any, priority int) {
	item.value = value
	item.priority = priority
	heap.Fix(pq, item.index)
}
