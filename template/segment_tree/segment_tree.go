package segment_tree

type segNode struct {
	left, right, val int
}

type SegmentTree struct {
	values []segNode
	// 要求操作满足区间可加性
	// 例如 + * | & ^ min max gcd mulMatrix 摩尔投票 最大子段和 ...
	op func(a, b int) int
}

func InitSegmentTree(arr []int, op func(a, b int) int) *SegmentTree {
	t := &SegmentTree{
		values: make([]segNode, len(arr)<<2),
		op:     op,
	}
	t.build(arr, 1, 1, len(arr))
	return t
}

// Update 范围 [0, n - 1]
func (t SegmentTree) Update(idx, val int) {
	t.update(1, idx+1, val)
}

// Query 闭区间 范围 [0, n - 1]
func (t SegmentTree) Query(left, right int) int {
	return t.query(1, left+1, right+1)
}

func (t SegmentTree) QueryAll() int {
	return t.values[1].val
}

func (t SegmentTree) build(arr []int, cur, left, right int) {
	t.values[cur].left, t.values[cur].right = left, right
	if left == right {
		t.values[cur].val = arr[left-1]
		return
	}
	mid := (left + right) >> 1
	t.build(arr, cur*2, left, mid)
	t.build(arr, cur*2+1, mid+1, right)
	t.values[cur].val = t.op(t.values[cur*2].val, t.values[cur*2+1].val)
}

func (t SegmentTree) update(cur, idx, val int) {
	if t.values[cur].left == t.values[cur].right {
		t.values[cur].val = val
		return
	}
	mid := (t.values[cur].left + t.values[cur].right) >> 1
	if mid >= idx {
		t.update(cur*2, idx, val)
	} else {
		t.update(cur*2+1, idx, val)
	}
	t.values[cur].val = t.op(t.values[cur*2].val, t.values[cur*2+1].val)
}

func (t SegmentTree) query(cur, queryLeft, queryRight int) int {
	if queryLeft <= t.values[cur].left && t.values[cur].right <= queryRight {
		return t.values[cur].val
	}
	mid := (t.values[cur].left + t.values[cur].right) >> 1
	if mid >= queryRight {
		return t.query(cur*2, queryLeft, queryRight)
	}
	if mid < queryLeft {
		return t.query(cur*2+1, queryLeft, queryRight)
	}
	vl, vr := t.query(cur*2, queryLeft, queryRight), t.query(cur*2+1, queryLeft, queryRight)
	return t.op(vl, vr)
}
