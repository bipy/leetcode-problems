package heaps

type heapData struct {
	data []interface{}
	less func(i, j interface{}) bool
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
