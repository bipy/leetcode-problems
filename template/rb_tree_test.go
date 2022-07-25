package template

import (
	"github.com/emirpasic/gods/trees/redblacktree"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestRedBlackTree_Size(t *testing.T) {
	rb := redblacktree.NewWithIntComparator()
	rb.Put(1, &Item{value: "hello"})
	rb.Put(2, &Item{value: "world"})
	assert.Equal(t, 2, rb.Size())
}

func TestRedBlackTree_Begin(t *testing.T) {
	rb := redblacktree.NewWithIntComparator()
	rb.Put(1, &Item{value: "hello"})
	rb.Put(2, &Item{value: "world"})
	assert.Equal(t, 1, rb.Left().Key.(int))
}

func TestRedBlackTree_End(t *testing.T) {
	rb := redblacktree.NewWithIntComparator()
	rb.Put(1, &Item{value: "hello"})
	rb.Put(2, &Item{value: "world"})
	assert.Equal(t, 2, rb.Right().Key.(int))
}

func TestRedBlackTree_Iterator(t *testing.T) {
	rb := redblacktree.NewWithIntComparator()
	rb.Put(1, &Item{value: "hello"})
	rb.Put(2, &Item{value: "world"})
	ans := ""
	it := rb.Iterator()
	for it.Next() {
		ans += it.Value().(*Item).value.(string) + " "
	}
	assert.Equal(t, "hello world ", ans)
}

func TestRedBlackTree_Get(t *testing.T) {
	rb := redblacktree.NewWithIntComparator()
	rb.Put(1, &Item{value: "hello"})
	rb.Put(2, &Item{value: "world"})
	n, ok := rb.Get(2)
	assert.True(t, ok)
	assert.Equal(t, "world", n.(*Item).value.(string))
}

func TestRedBlackTree_Floor(t *testing.T) {
	rb := redblacktree.NewWithIntComparator()
	rb.Put(1, &Item{value: "hello"})
	rb.Put(2, &Item{value: "world"})
	// 闭区间
	n, ok := rb.Floor(3333)
	assert.True(t, ok)
	assert.Equal(t, "world", n.Value.(*Item).value.(string))
	_, ok = rb.Floor(0)
	assert.False(t, ok)
}

func TestRedBlackTree_Ceiling(t *testing.T) {
	rb := redblacktree.NewWithIntComparator()
	rb.Put(1, &Item{value: "hello"})
	rb.Put(2, &Item{value: "world"})
	// 闭区间
	n, ok := rb.Ceiling(1)
	assert.True(t, ok)
	assert.Equal(t, "hello", n.Value.(*Item).value.(string))
	_, ok = rb.Ceiling(100)
	assert.False(t, ok)
}
