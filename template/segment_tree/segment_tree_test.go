package segment_tree

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSegmentTree(t *testing.T) {
	st := InitSegmentTree(make([]int, 10))
	assert.Equal(t, 0, st.QuerySum(1, 10))
	st.Add(2, 5)
	st.Add(3, 6)
	st.Add(7, 10)
	assert.Equal(t, 21, st.QuerySum(1, 10))
}
