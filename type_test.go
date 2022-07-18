package leetcode

import (
	"fmt"
	"testing"
)

func TestListNode_Show(t *testing.T) {
	list := ListDeserialize("[1,2,3,4,5,6,7,8]")
	list.Show()
}

func TestTreeNode_Show(t *testing.T) {
	tree := TreeDeserialize("[1,2,3,4,null,5]")
	tree.Show()
}

func TestCntMap(t *testing.T) {
	cm := cntMap{}
	s := "hellooooworld"
	fmt.Println("Add")
	for i := range s {
		cm.Add(s[i])
		fmt.Println(cm.Cnt, "Len: ", cm.Len)
	}
	fmt.Println("Remove")
	for i := range s {
		cm.Remove(s[i])
		fmt.Println(cm.Cnt, "Len: ", cm.Len)
	}
}
