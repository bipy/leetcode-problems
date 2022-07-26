package deque

type Item struct {
	value    interface{}
	priority int
	index    int
}

type Deque struct {
	left, right []*Item
}

func (d Deque) Len() int {
	return len(d.right) + len(d.left)
}

func (d Deque) Less(i, j int) bool {
	return d.Get(i).priority < d.Get(j).priority
}

func (d Deque) Swap(i, j int) {
	ii, jj := d.getPtr(i), d.getPtr(j)
	*ii, *jj = *jj, *ii
}

func (d Deque) Empty() bool {
	return d.Len() == 0
}

func (d *Deque) PushBack(item *Item) {
	d.right = append(d.right, item)
}

func (d *Deque) PushFront(item *Item) {
	d.left = append(d.left, item)
}

func (d *Deque) PopBack() {
	if len(d.right) > 0 {
		d.right = d.right[:len(d.right)-1]
	} else if len(d.left) > 0 {
		d.left = d.left[1:]
	}
}

func (d *Deque) PopFront() {
	if len(d.left) > 0 {
		d.left = d.left[:len(d.left)-1]
	} else if len(d.right) > 0 {
		d.right = d.right[1:]
	}
}

func (d Deque) Back() *Item {
	return d.Get(d.Len() - 1)
}

func (d Deque) Front() *Item {
	return d.Get(0)
}

func (d Deque) Get(i int) *Item {
	ll := len(d.left)
	if i < ll {
		return d.left[ll-i-1]
	}
	return d.right[i-ll]
}

func (d Deque) getPtr(i int) **Item {
	ll := len(d.left)
	if i < ll {
		return &d.left[ll-i-1]
	}
	return &d.right[i-ll]
}
