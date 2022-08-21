package heaps

import (
	"container/heap"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestHeap(t *testing.T) {
	type node struct {
		key, val int
	}

	nodes := []node{{5, 2}, {2, 2}, {3, 2}, {2, 5}}

	trans := func() []interface{} {
		rt := make([]interface{}, len(nodes))
		for i := range rt {
			rt[i] = nodes[i]
		}
		return rt
	}

	h := InitHeap(trans(), func(i, j interface{}) bool {
		if i.(node).key == j.(node).key {
			return i.(node).val > j.(node).val
		}
		return i.(node).key < j.(node).key
	})

	assert.Equal(t, node{2, 5}, h.Top())
	assert.Equal(t, node{2, 5}, heap.Pop(h))
	assert.Equal(t, node{2, 2}, heap.Pop(h))

	heap.Push(h, node{2, 1})
	assert.Equal(t, node{2, 1}, h.Top())

	h.Update(0, node{3, 1})
	assert.Equal(t, node{3, 2}, h.Top())

}
