package template

type UnionFind struct {
	Father []int
	Size   []int
	Groups int
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
	if x != uf.Father[x] {
		uf.Father[x] = uf.Find(uf.Father[x])
	}
	return uf.Father[x]
}

func (uf *UnionFind) Union(x, y int) bool {
	x = uf.Find(x)
	y = uf.Find(y)
	if x == y {
		return false
	}
	if uf.Size[x] > uf.Size[y] {
		x, y = y, x
	}
	uf.Father[x] = y
	uf.Size[y] += uf.Size[x]
	uf.Groups--
	return true
}
