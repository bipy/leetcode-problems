package main

import (
	"fmt"
	"strconv"
	"strings"
)

const mod = 1e9 + 7

var directions = []struct{ x, y int }{{1, 0}, {0, -1}, {-1, 0}, {0, 1}}

// TreeSerialize Serializes a tree to a single string.
func TreeSerialize(root *TreeNode) string {
	if root == nil {
		return ""
	}
	sb := strings.Builder{}
	var queue []*TreeNode
	queue = append(queue, root)
	for len(queue) != 0 {
		n := len(queue)
		flag := false
		for i := 0; i < n; i++ {
			if queue[i] != nil {
				flag = true
				break
			}
		}
		if !flag {
			break
		}
		for i := 0; i < n; i++ {
			cur := queue[i]
			if cur == nil {
				queue = append(queue, nil, nil)
				sb.WriteString("null")
			} else {
				queue = append(queue, cur.Left, cur.Right)
				sb.WriteString(strconv.Itoa(cur.Val))
			}
			sb.WriteByte(',')
		}
		queue = queue[n:]
	}
	return strings.TrimSuffix(sb.String(), ",")
}

// TreeDeserialize Deserializes your encoded data to tree.
func TreeDeserialize(data string) *TreeNode {
	if data == "" {
		return nil
	}
	nodes := strings.Split(data, ",")
	var insert func(int) *TreeNode
	insert = func(idx int) *TreeNode {
		if idx >= len(nodes) || nodes[idx] == "null" {
			return nil
		}
		val, _ := strconv.Atoi(nodes[idx])
		return &TreeNode{
			Val:   val,
			Left:  insert(idx<<1 + 1),
			Right: insert(idx<<1 + 2),
		}
	}
	return insert(0)
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func max(x, y int) int {
	if x < y {
		return y
	}
	return x
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func minIdx(nums ...int) (idx int) {
	for i, v := range nums {
		if nums[idx] > v {
			idx = i
		}
	}
	return idx
}

func maxIdx(nums ...int) (idx int) {
	for i, v := range nums {
		if nums[idx] < v {
			idx = i
		}
	}
	return idx
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

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func StrToStrSlice(input string) []string {
	input = strings.TrimPrefix(input, "[")
	input = strings.TrimSuffix(input, "]")
	if input == "" {
		return nil
	}
	rt := strings.Split(input, ",")
	for i := range rt {
		rt[i] = strings.TrimPrefix(rt[i], "\"")
		rt[i] = strings.TrimSuffix(rt[i], "\"")
	}
	return rt
}

func StrToIntSlice(input string) (rt []int) {
	input = strings.TrimPrefix(input, "[")
	input = strings.TrimSuffix(input, "]")
	if input == "" {
		return nil
	}
	s := strings.Split(input, ",")
	for i := range s {
		d, err := strconv.Atoi(s[i])
		if err != nil {
			fmt.Println("Error")
		}
		rt = append(rt, d)
	}
	return
}

func StrTo2DIntSlice(input string) (rt [][]int) {
	input = strings.TrimPrefix(input, "[[")
	input = strings.TrimSuffix(input, "]]")
	s := strings.Split(input, "],[")
	for i := range s {
		if is := StrToIntSlice(s[i]); is != nil {
			rt = append(rt, is)
		}
	}
	return
}

func StrTo2DStrSlice(input string) (rt [][]string) {
	input = strings.TrimPrefix(input, "[[")
	input = strings.TrimSuffix(input, "]]")
	s := strings.Split(input, "],[")
	for i := range s {
		if ss := StrToStrSlice(s[i]); ss != nil {
			rt = append(rt, ss)
		}
	}
	return
}
