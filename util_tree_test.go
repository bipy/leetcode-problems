package leetcode

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestTreeSerialize(t *testing.T) {
	type testCase struct {
		str  string
		tree *TreeNode
	}
	tests := []testCase{
		{
			str: "[1,2,4]",
			tree: &TreeNode{
				Val:   1,
				Left:  &TreeNode{Val: 2},
				Right: &TreeNode{Val: 4},
			},
		},
		{
			str: "[1,2,null,4]",
			tree: &TreeNode{
				Val: 1,
				Left: &TreeNode{
					Val:  2,
					Left: &TreeNode{Val: 4},
				},
			},
		},
		{
			str: "[1,3,4,5,1]",
			tree: &TreeNode{
				Val: 1,
				Left: &TreeNode{
					Val:   3,
					Left:  &TreeNode{Val: 5},
					Right: &TreeNode{Val: 1},
				},
				Right: &TreeNode{Val: 4},
			},
		},
		{
			str: "[1,2,3,4,5,null,null,6]",
			tree: &TreeNode{
				Val: 1,
				Left: &TreeNode{
					Val: 2,
					Left: &TreeNode{
						Val:  4,
						Left: &TreeNode{Val: 6},
					},
					Right: &TreeNode{Val: 5},
				},
				Right: &TreeNode{Val: 3},
			},
		},
		{
			str:  "[]",
			tree: nil,
		},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.str, TreeSerialize(tt.tree))
	}
}

func TestTreeDeserialize(t *testing.T) {
	type testCase struct {
		str  string
		tree *TreeNode
	}
	tests := []testCase{
		{
			str: "[1,2,4]",
			tree: &TreeNode{
				Val:   1,
				Left:  &TreeNode{Val: 2},
				Right: &TreeNode{Val: 4},
			},
		},
		{
			str: "[1,2,null,4]",
			tree: &TreeNode{
				Val: 1,
				Left: &TreeNode{
					Val:  2,
					Left: &TreeNode{Val: 4},
				},
			},
		},
		{
			str: "[1,3,4,5,1]",
			tree: &TreeNode{
				Val: 1,
				Left: &TreeNode{
					Val:   3,
					Left:  &TreeNode{Val: 5},
					Right: &TreeNode{Val: 1},
				},
				Right: &TreeNode{Val: 4},
			},
		},
		{
			str: "[1,2,3,4,5,null,null,6]",
			tree: &TreeNode{
				Val: 1,
				Left: &TreeNode{
					Val: 2,
					Left: &TreeNode{
						Val:  4,
						Left: &TreeNode{Val: 6},
					},
					Right: &TreeNode{Val: 5},
				},
				Right: &TreeNode{Val: 3},
			},
		},
		{
			str:  "[]",
			tree: nil,
		},
	}
	for _, tt := range tests {
		assert.True(t, TreeEqual(tt.tree, TreeDeserialize(tt.str)))
	}
}
