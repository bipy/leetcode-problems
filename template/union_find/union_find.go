package union_find

type UnionFind struct {
	father []int
	size   []int
	groups int
}

func InitUnionFind(n int) *UnionFind {
	fa := make([]int, n)
	sz := make([]int, n)
	for i := range fa {
		fa[i] = i
		sz[i] = 1
	}
	return &UnionFind{fa, sz, n}
}

func (uf *UnionFind) Find(x int) int {
	if x != uf.father[x] {
		uf.father[x] = uf.Find(uf.father[x])
	}
	return uf.father[x]
}

func (uf *UnionFind) Union(x, y int) bool {
	x = uf.Find(x)
	y = uf.Find(y)
	if x == y {
		return false
	}
	if uf.size[x] > uf.size[y] {
		x, y = y, x
	}
	uf.father[x] = y
	uf.size[y] += uf.size[x]
	uf.groups--
	return true
}

func (uf UnionFind) Groups() int {
	return uf.groups
}

func (uf UnionFind) GroupSize(i int) int {
	return uf.size[uf.Find(i)]
}
