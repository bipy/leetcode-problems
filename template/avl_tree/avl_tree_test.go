package avl_tree

import (
	"github.com/stretchr/testify/assert"
	"strconv"
	"testing"
)

func TestAVLTree(t *testing.T) {
	tree := InitAVLTree(func(i, j interface{}) bool {
		return i.(int) < j.(int)
	})
	for i := 0; i < 10000; i++ {
		tree.Put(i, strconv.Itoa(i))
	}
	if v, ok := tree.Get(3333); ok {
		assert.EqualValues(t, "3333", v)
	}
}
