package deque

type Deque struct {
	left, right []interface{}
	less        func(a, b interface{}) bool
}

func InitDeque(less func(a, b interface{}) bool) *Deque {
	return &Deque{less: less}
}

func (d Deque) Len() int {
	return len(d.right) + len(d.left)
}

func (d Deque) Less(i, j int) bool {
	return d.less(d.Get(i), d.Get(j))
}

func (d Deque) Swap(i, j int) {
	ii, jj := d.Get(i), d.Get(j)
	d.Set(i, jj)
	d.Set(j, ii)
}

func (d *Deque) PushBack(x interface{}) {
	d.right = append(d.right, x)
}

func (d *Deque) PushFront(x interface{}) {
	d.left = append(d.left, x)
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

func (d Deque) Back() interface{} {
	return d.Get(d.Len() - 1)
}

func (d Deque) Front() interface{} {
	return d.Get(0)
}

func (d Deque) Get(i int) interface{} {
	ll := len(d.left)
	if i < ll {
		return d.left[ll-i-1]
	}
	return d.right[i-ll]
}

func (d Deque) Set(i int, x interface{}) {
	ll := len(d.left)
	if i < ll {
		d.left[ll-i-1] = x
	} else {
		d.right[i-ll] = x
	}
}
