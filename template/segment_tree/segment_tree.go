package segment_tree

type SegmentTree struct {
	arr  []int
	sum  []int
	root int
	size int
}

func InitSegmentTree(arr []int) *SegmentTree {
	t := &SegmentTree{
		arr:  arr,
		sum:  make([]int, len(arr)<<2),
		root: 1,
		size: len(arr),
	}
	t.build(1, t.root, t.size)
	return t
}

func (t SegmentTree) Add(idx, val int) {
	t.add(1, t.root, t.size, idx, val)
}

func (t SegmentTree) QuerySum(left, right int) int {
	return t.querySum(1, t.root, t.size, left, right)
}

func (t SegmentTree) build(cur, left, right int) {
	if left == right {
		t.sum[cur] = t.arr[left-1]
		return
	}
	mid := (left + right) >> 1
	t.build(cur*2, left, mid)
	t.build(cur*2+1, mid+1, right)
	t.sum[cur] = t.sum[cur*2] + t.sum[cur*2+1]
}

func (t SegmentTree) add(cur, left, right, idx, val int) {
	if left == right {
		t.sum[cur] += val
		return
	}
	mid := (left + right) >> 1
	if mid >= idx {
		t.add(cur*2, left, mid, idx, val)
	} else {
		t.add(cur*2+1, mid+1, right, idx, val)
	}
	t.sum[cur] = t.sum[cur*2] + t.sum[cur*2+1]
}

func (t SegmentTree) querySum(cur, left, right, queryLeft, queryRight int) int {
	if left <= queryLeft && right >= queryRight {
		return t.sum[cur]
	}
	sum := 0
	mid := (left + right) >> 1
	if mid >= queryLeft {
		sum += t.querySum(cur*2, left, mid, queryLeft, queryRight)
	}
	if mid < queryRight {
		sum += t.querySum(cur*2+1, mid+1, right, queryLeft, queryRight)
	}
	return sum
}
