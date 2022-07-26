package skip_list

import (
	"math/rand"
)

const p = 0.5

const MaxLevel = 32

type node struct {
	forward    []*node
	key, value interface{}
}

type SkipList struct {
	compare func(i, j interface{}) int
	head    *node
	length  int
	level   int
}

func (s SkipList) Len() int {
	return s.length
}

func InitSkipList(comparator func(i, j interface{}) int) *SkipList {
	return &SkipList{
		compare: comparator,
		head:    &node{forward: make([]*node, MaxLevel)},
		length:  0,
		level:   0,
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
		for cur.forward[i] != nil && s.compare(cur.forward[i].key, key) < 0 {
			cur = cur.forward[i]
		}
	}
	cur = cur.forward[0]
	if cur != nil && s.compare(cur.key, key) == 0 {
		return cur.value, true
	}
	return nil, false
}

func (s *SkipList) Insert(key, value interface{}) {
	update := make([]*node, MaxLevel)
	for i := range update {
		update[i] = s.head
	}
	cur := s.head
	for i := s.level - 1; i >= 0; i-- {
		// 找到第 i 层小于且最接近 num 的元素
		for cur.forward[i] != nil && s.compare(cur.forward[i].key, key) < 0 {
			cur = cur.forward[i]
		}
		update[i] = cur
	}
	lv := s.randomLevel()
	if lv > s.level {
		s.level = lv
	}
	nn := &node{
		forward: make([]*node, lv),
		key:     key,
		value:   value,
	}
	for i := range update[:lv] {
		nn.forward[i] = update[i].forward[i]
		update[i].forward[i] = nn
	}
}

func (s *SkipList) Remove(key interface{}) bool {
	update := make([]*node, MaxLevel)
	cur := s.head
	for i := s.level - 1; i >= 0; i-- {
		// 找到第 i 层小于且最接近 num 的元素
		for cur.forward[i] != nil && s.compare(cur.forward[i].key, key) < 0 {
			cur = cur.forward[i]
		}
		update[i] = cur
	}
	cur = cur.forward[0]
	// 如果值不存在则返回 false
	if cur == nil || s.compare(cur.key, key) != 0 {
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
	return true
}

func IntComparator(i, j interface{}) int {
	return i.(int) - j.(int)
}

func StringComparator(i, j interface{}) int {
	if i.(string) < j.(string) {
		return -1
	}
	if i.(string) > j.(string) {
		return 1
	}
	return 0
}
