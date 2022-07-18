package leetcode

import "fmt"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func (n *TreeNode) Show() {
	fmt.Println(TreeSerialize(n))
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func (n *ListNode) Show() {
	fmt.Printf("%d->", n.Val)
	for p := n.Next; p != nil && p != n; p = p.Next {
		fmt.Printf("%d->", p.Val)
	}
	fmt.Println("nil")
}

type cntMap struct {
	Cnt [26]int
	Len int
}

func (m *cntMap) Add(c byte) {
	if m.Cnt[c-'a'] == 0 {
		m.Len++
	}
	m.Cnt[c-'a']++
}

func (m *cntMap) Remove(c byte) {
	if m.Cnt[c-'a'] == 1 {
		m.Len--
	}
	m.Cnt[c-'a']--
}
