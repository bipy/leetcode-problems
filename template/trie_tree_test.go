package template

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestTrieTree(t *testing.T) {
	tree := InitTrieTree()
	tree.Insert("hello")
	tree.Insert("world")
	tree.Insert("haha")
	assert.True(t, tree.Query("hello"))
	assert.True(t, tree.Query("world"))
	assert.True(t, tree.Query("haha"))
	assert.False(t, tree.Query("my"))
	assert.False(t, tree.Query("hallo"))
	assert.False(t, tree.Query("hellohaha"))
	assert.False(t, tree.Query("ha"))
	assert.False(t, tree.Query("worldd"))
}
