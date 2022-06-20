package template

// Heap min heap
type node struct {
	x, y int
}

type Heap []node

func (h Heap) Len() int {
	return len(h)
}

func (h Heap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h Heap) Less(i, j int) bool {
	return h[i].y < h[j].y
}

func (h *Heap) Push(x any) {
	*h = append(*h, x.(node))
}

func (h *Heap) Pop() any {
	n := len(*h)
	x := (*h)[n-1]
	*h = (*h)[:n-1]
	return x
}
