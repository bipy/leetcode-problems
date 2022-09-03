package sparse_table

import "math/bits"

type SparseTable struct {
	data [][]int
	op   func(a, b int) int
}

// InitSparseTable op: (min, max, gcd)
func InitSparseTable(arr []int, op func(a, b int) int) *SparseTable {
	n := len(arr)
	sz := bits.Len(uint(n))
	st := make([][]int, n)
	for i, v := range arr {
		st[i] = make([]int, sz)
		st[i][0] = v
	}
	for j := 1; 1<<j <= n; j++ {
		for i := 0; i+1<<j <= n; i++ {
			st[i][j] = op(st[i][j-1], st[i+1<<(j-1)][j-1])
		}
	}
	return &SparseTable{
		data: st,
		op:   op,
	}
}

// Query [l, r) 前闭后开 0开始
func (st SparseTable) Query(left, right int) int {
	k := bits.Len(uint(right-left)) - 1
	return st.op(st.data[left][k], st.data[right-1<<k][k])
}
