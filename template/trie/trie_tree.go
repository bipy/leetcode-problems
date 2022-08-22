package trie

type trieNode struct {
	next [26]*trieNode
	end  bool
}

type Trie struct {
	*trieNode
}

func InitTrie() *Trie {
	return &Trie{&trieNode{}}
}

func (t Trie) Insert(s string) {
	p := t.trieNode
	for _, c := range s {
		if p.next[c-'a'] == nil {
			p.next[c-'a'] = &trieNode{}
		}
		p = p.next[c-'a']
	}
	p.end = true
}

func (t Trie) Query(s string) bool {
	p := t.trieNode
	for _, c := range s {
		if p.next[c-'a'] == nil {
			return false
		}
		p = p.next[c-'a']
	}
	return p.end
}

func (t Trie) QueryPrefix(s string) bool {
	p := t.trieNode
	for _, c := range s {
		if p.next[c-'a'] == nil {
			return false
		}
		p = p.next[c-'a']
	}
	return true
}
