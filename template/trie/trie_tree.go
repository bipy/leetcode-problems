package trie

type node struct {
	next [26]*node
	end  bool
}

type Trie struct {
	*node
}

func InitTrie() *Trie {
	return &Trie{&node{}}
}

func (t Trie) Insert(s string) {
	p := t.node
	for _, c := range s {
		if p.next[c-'a'] == nil {
			p.next[c-'a'] = &node{}
		}
		p = p.next[c-'a']
	}
	p.end = true
}

func (t Trie) Query(s string) bool {
	p := t.node
	for _, c := range s {
		if p.next[c-'a'] == nil {
			return false
		}
		p = p.next[c-'a']
	}
	return p.end
}

func (t Trie) QueryPrefix(s string) bool {
	p := t.node
	for _, c := range s {
		if p.next[c-'a'] == nil {
			return false
		}
		p = p.next[c-'a']
	}
	return true
}
