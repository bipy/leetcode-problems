package deque

import (
	"github.com/stretchr/testify/assert"
	"sort"
	"testing"
)

func TestDeque(t *testing.T) {
	dq := InitDeque(func(a, b interface{}) bool {
		return a.(int) < b.(int)
	})
	dq.PushBack(1)
	// 1

	assert.Equal(t, 1, dq.Front().(int))
	assert.Equal(t, 1, dq.Back().(int))

	dq.PushBack(2)
	// 1 2
	dq.PushFront(3)
	// 3 1 2

	assert.Equal(t, 3, dq.Len())
	assert.Equal(t, 3, dq.Front().(int))
	assert.Equal(t, 2, dq.Back().(int))

	sort.Sort(dq)
	// 1 2 3

	assert.Equal(t, 1, dq.Front().(int))
	assert.Equal(t, 3, dq.Back().(int))

	dq.PopBack()
	// 1 2

	assert.Equal(t, 2, dq.Back().(int))

	dq.PopFront()
	// 2

	assert.Equal(t, 2, dq.Front().(int))
	assert.Equal(t, 1, dq.Len())

	dq.PopFront()

	assert.True(t, dq.Len() == 0)
}
