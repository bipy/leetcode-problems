package leetcode

import (
	"strconv"
	"strings"
)

func ListDeserialize(data string) *ListNode {
	data = strings.TrimSuffix(data, "]")
	data = strings.TrimPrefix(data, "[")
	if data == "" {
		return nil
	}
	nodes := strings.Split(data, ",")
	H := &ListNode{}
	p := H
	for i := range nodes {
		v, err := strconv.Atoi(nodes[i])
		if err != nil {
			panic(err.Error())
		}
		p.Next = &ListNode{Val: v}
		p = p.Next
	}
	return H.Next
}

func CycleListDeserialize(data string) *ListNode {
	data = strings.TrimSuffix(data, "]")
	data = strings.TrimPrefix(data, "[")
	if data == "" {
		return nil
	}
	nodes := strings.Split(data, ",")
	H := &ListNode{}
	p := H
	for i := range nodes {
		v, err := strconv.Atoi(nodes[i])
		if err != nil {
			panic(err.Error())
		}
		p.Next = &ListNode{Val: v}
		p = p.Next
	}
	p.Next = H.Next
	return H.Next
}

func ListEqual(l1, l2 *ListNode) bool {
	p, q := l1, l2
	for p != nil && q != nil {
		if p.Val != q.Val {
			return false
		}
		p, q = p.Next, q.Next
		if p == l1 && q == l2 {
			return true
		}
	}
	return q == nil && p == nil
}
