package skip_list

import (
	"math/rand"
)

const p = 0.5

const MaxLevel = 32

type Node struct {
	forward    []*Node
	Key, Value interface{}
}

func (n Node) Next() *Node {
	if len(n.forward) == 0 {
		return nil
	}
	return n.forward[0]
}

type SkipList struct {
	less   func(i, j interface{}) bool
	head   *Node
	length int
	level  int
}

func (s SkipList) Len() int {
	return s.length
}

func (s SkipList) equal(i, j interface{}) bool {
	return !s.less(i, j) && !s.less(j, i)
}

func InitSkipList(less func(i, j interface{}) bool) *SkipList {
	return &SkipList{
		less:   less,
		head:   &Node{forward: make([]*Node, MaxLevel)},
		length: 0,
		level:  0,
	}
}

func (SkipList) randomLevel() int {
	lv := 1
	for lv < MaxLevel && rand.Float64() < p {
		lv++
	}
	return lv
}

func (s *SkipList) Find(key interface{}) (interface{}, bool) {
	cur := s.head
	for i := s.level - 1; i >= 0; i-- {
		// 找到第 i 层小于且最接近 target 的元素
		for cur.forward[i] != nil && s.less(cur.forward[i].Key, key) {
			cur = cur.forward[i]
		}
		if cur.forward[i] != nil && s.equal(cur.forward[i].Key, key) {
			return cur.Value, true
		}
	}
	return nil, false
}

func (s *SkipList) Insert(key, value interface{}) {
	update := make([]*Node, MaxLevel)
	for i := range update {
		update[i] = s.head
	}
	cur := s.head
	for i := s.level - 1; i >= 0; i-- {
		// 找到第 i 层小于且最接近 num 的元素
		for cur.forward[i] != nil && s.less(cur.forward[i].Key, key) {
			cur = cur.forward[i]
		}
		update[i] = cur
	}
	lv := s.randomLevel()
	if lv > s.level {
		s.level = lv
	}
	nn := &Node{
		forward: make([]*Node, lv),
		Key:     key,
		Value:   value,
	}
	for i := range update[:lv] {
		nn.forward[i] = update[i].forward[i]
		update[i].forward[i] = nn
	}
	s.length++
}

func (s *SkipList) Remove(key interface{}) bool {
	update := make([]*Node, MaxLevel)
	cur := s.head
	for i := s.level - 1; i >= 0; i-- {
		// 找到第 i 层小于且最接近 num 的元素
		for cur.forward[i] != nil && s.less(cur.forward[i].Key, key) {
			cur = cur.forward[i]
		}
		update[i] = cur
	}
	cur = cur.forward[0]
	// 如果值不存在则返回 false
	if cur == nil || !s.equal(cur.Key, key) {
		return false
	}
	for i := 0; i < s.level && update[i].forward[i] == cur; i++ {
		// 对第 i 层的状态进行更新，将 forward 指向被删除节点的下一跳
		update[i].forward[i] = cur.forward[i]
	}
	// 更新当前的 level
	for s.level > 1 && s.head.forward[s.level-1] == nil {
		s.level--
	}
	s.length--
	return true
}

func (s *SkipList) Left() *Node {
	return s.head.forward[0]
}

func (s *SkipList) Right() *Node {
	cur := s.head
	for i := s.level - 1; i >= 0; i-- {
		for cur.forward[i] != nil {
			cur = cur.forward[i]
		}
	}
	return cur
}
