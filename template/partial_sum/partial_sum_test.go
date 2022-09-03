package partial_sum

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestPartialSum(t *testing.T) {
	arr := []int{12, 2, 4, 12, 45}
	ps := InitPartialSum(arr)
	assert.Equal(t, 6, ps.Query(1, 2))
	assert.Equal(t, 18, ps.Query(0, 2))
	assert.Equal(t, 75, ps.Query(0, 4))
	assert.Equal(t, 75, ps.QueryAll())
}
