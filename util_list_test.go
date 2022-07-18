package leetcode

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestListDeserialize(t *testing.T) {
	type testCase struct {
		str  string
		list *ListNode
	}
	tests := []testCase{
		{
			str: "[1,2,4]",
			list: &ListNode{
				Val: 1,
				Next: &ListNode{
					Val: 2,
					Next: &ListNode{
						Val:  4,
						Next: nil,
					},
				},
			},
		},
		{
			str:  "[]",
			list: nil,
		},
	}
	for _, tt := range tests {
		assert.True(t, ListEqual(tt.list, ListDeserialize(tt.str)))
	}
}

func TestCycleListDeserialize(t *testing.T) {
	type testCase struct {
		str  string
		list *ListNode
	}
	tests := []testCase{
		{
			str: "[1,2,4]",
			list: &ListNode{
				Val: 1,
				Next: &ListNode{
					Val: 2,
					Next: &ListNode{
						Val:  4,
						Next: nil,
					},
				},
			},
		},
		{
			str:  "[]",
			list: nil,
		},
	}
	for i := range tests {
		p := tests[i].list
		for p != nil {
			if p.Next == nil {
				p.Next = tests[i].list
				break
			}
			p = p.Next
		}
	}
	for _, tt := range tests {
		assert.True(t, ListEqual(tt.list, CycleListDeserialize(tt.str)))
	}
}
