package template

type TrieNode struct {
	next [26]*TrieNode
	end  bool
}

type TrieTree struct {
	*TrieNode
}

func InitTrieTree() *TrieTree {
	return &TrieTree{&TrieNode{}}
}

func (t TrieTree) Insert(s string) {
	p := t.TrieNode
	for _, c := range s {
		if p.next[c-'a'] == nil {
			p.next[c-'a'] = &TrieNode{}
		}
		p = p.next[c-'a']
	}
	p.end = true
}

func (t TrieTree) Query(s string) bool {
	p := t.TrieNode
	for _, c := range s {
		if p.next[c-'a'] == nil {
			return false
		}
		p = p.next[c-'a']
	}
	return p.end
}

func (t TrieTree) QueryPrefix(s string) bool {
	p := t.TrieNode
	for _, c := range s {
		if p.next[c-'a'] == nil {
			return false
		}
		p = p.next[c-'a']
	}
	return true
}
