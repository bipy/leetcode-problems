package leetcode

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func (n *ListNode) Show() {
	if n == nil {
		fmt.Println("nil")
		return
	}
	fmt.Printf("%d->", n.Val)
	for p := n.Next; p != nil && p != n; p = p.Next {
		fmt.Printf("%d->", p.Val)
	}
}

type cntMap struct {
	cnt [26]int
	len int
}

func (m *cntMap) add(c byte) {
	if m.cnt[c-'a'] == 0 {
		m.len++
	}
	m.cnt[c-'a']++
}

func (m *cntMap) remove(c byte) {
	if m.cnt[c-'a'] == 1 {
		m.len--
	}
	m.cnt[c-'a']--
}
