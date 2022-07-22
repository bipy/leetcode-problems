package template

import (
	"github.com/stretchr/testify/assert"
	"sort"
	"testing"
)

func TestDeque(t *testing.T) {
	dq := &Deque{}
	dq.PushBack(&Item{value: 1, priority: 1})
	// 1

	assert.Equal(t, 1, dq.Front().value.(int))
	assert.Equal(t, 1, dq.Back().value.(int))

	dq.PushBack(&Item{value: 2, priority: 2})
	// 1 2
	dq.PushFront(&Item{value: 3, priority: 3})
	// 3 1 2

	assert.Equal(t, 3, dq.Len())
	assert.Equal(t, 3, dq.Front().value.(int))
	assert.Equal(t, 2, dq.Back().value.(int))

	sort.Sort(dq)
	// 1 2 3

	assert.Equal(t, 1, dq.Front().value.(int))
	assert.Equal(t, 3, dq.Back().value.(int))

	dq.PopBack()
	// 1 2

	assert.Equal(t, 2, dq.Back().value.(int))

	dq.PopFront()
	// 2

	assert.Equal(t, 2, dq.Front().value.(int))
	assert.False(t, dq.Empty())

	dq.PopFront()

	assert.True(t, dq.Empty())
}
