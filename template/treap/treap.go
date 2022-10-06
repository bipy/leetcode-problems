package treap

import "time"

type node struct {
	lr       [2]*node
	priority uint
	key      int
	dupCnt   int
	sz       int
}

func (o *node) cmp(a int) int {
	b := o.key
	if a == b {
		return -1
	}
	if a < b {
		return 0
	}
	return 1
}

func (o *node) size() int {
	if o != nil {
		return o.sz
	}
	return 0
}

func (o *node) maintain() {
	o.sz = o.dupCnt + o.lr[0].size() + o.lr[1].size()
}

func (o *node) rotate(d int) *node {
	x := o.lr[d^1]
	o.lr[d^1] = x.lr[d]
	x.lr[d] = o
	o.maintain()
	x.maintain()
	return x
}

type Treap struct {
	rd   uint
	root *node
}

func (t *Treap) fastRand() uint {
	t.rd ^= t.rd << 13
	t.rd ^= t.rd >> 17
	t.rd ^= t.rd << 5
	return t.rd
}

func (t *Treap) put(o *node, key int) *node {
	if o == nil {
		return &node{priority: t.fastRand(), key: key, dupCnt: 1, sz: 1}
	}
	if d := o.cmp(key); d >= 0 {
		o.lr[d] = t.put(o.lr[d], key)
		if o.lr[d].priority > o.priority {
			o = o.rotate(d ^ 1)
		}
	} else {
		o.dupCnt++
	}
	o.maintain()
	return o
}

func (t *Treap) Put(key int) {
	t.root = t.put(t.root, key)
}

func (t *Treap) delete(o *node, key int) *node {
	if o == nil {
		return nil
	}
	if d := o.cmp(key); d >= 0 {
		o.lr[d] = t.delete(o.lr[d], key)
	} else {
		if o.dupCnt > 1 {
			o.dupCnt--
		} else {
			if o.lr[1] == nil {
				return o.lr[0]
			}
			if o.lr[0] == nil {
				return o.lr[1]
			}
			d = 0
			if o.lr[0].priority > o.lr[1].priority {
				d = 1
			}
			o = o.rotate(d)
			o.lr[d] = t.delete(o.lr[d], key)
		}
	}
	o.maintain()
	return o
}

func (t *Treap) Delete(key int) {
	t.root = t.delete(t.root, key)
}

func InitTreap() *Treap {
	return &Treap{
		rd: uint(time.Now().UnixNano())/2 + 1,
	}
}

// Rank < key 的元素个数
func (t *Treap) Rank(key int) (kth int) {
	for o := t.root; o != nil; {
		switch c := o.cmp(key); {
		case c == 0:
			o = o.lr[0]
		case c > 0:
			kth += o.lr[0].size() + o.dupCnt
			o = o.lr[1]
		default:
			kth += o.lr[0].size()
			return
		}
	}
	return
}
