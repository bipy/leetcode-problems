package union_find

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestUnionFind(t *testing.T) {
	uf := InitUnionFind(10)
	assert.Equal(t, 1, uf.Find(1))
	assert.Equal(t, 7, uf.Find(7))
	assert.True(t, uf.Union(1, 7))
	assert.True(t, uf.Union(1, 8))
	assert.True(t, uf.Union(2, 7))
	assert.Equal(t, 4, uf.GroupSize(1))
	assert.False(t, uf.Union(1, 8))
	assert.False(t, uf.Union(2, 7))
	assert.True(t, uf.Union(3, 2))
	assert.False(t, uf.Union(2, 1))
}
