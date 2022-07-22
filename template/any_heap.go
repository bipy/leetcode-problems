package template

// Heap min-heap
type Heap []*Item

func (h Heap) Len() int {
	return len(h)
}

func (h Heap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h Heap) Less(i, j int) bool {
	return h[i].priority < h[j].priority
}

func (h *Heap) Push(x any) {
	*h = append(*h, x.(*Item))
}

func (h *Heap) Pop() any {
	n := len(*h)
	x := (*h)[n-1]
	*h = (*h)[:n-1]
	return x
}
