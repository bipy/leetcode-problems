package leetcode

import (
	"strconv"
	"strings"
)

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
	data = strings.TrimPrefix(data, "[")
	data = strings.TrimSuffix(data, "]")
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
