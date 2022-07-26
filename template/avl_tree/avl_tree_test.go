package avl_tree

import (
	"github.com/stretchr/testify/assert"
	"strconv"
	"testing"
)

func TestAVLTree(t *testing.T) {
	tree := AVLTree{}
	for i := 0; i < 10000; i++ {
		tree.Put(i, strconv.Itoa(i))
	}
	if v, ok := tree.Find(3333); ok {
		assert.EqualValues(t, "3333", v)
	}
}
